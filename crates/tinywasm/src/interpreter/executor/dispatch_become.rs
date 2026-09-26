use super::*;

struct Unbudgeted;
struct Bounded;

type UnbudgetedHandler = for<'store, 'module> fn(
    &mut Executor<'store, 'module>,
    &[Instruction],
    FuncAddr,
    usize,
    Instruction,
) -> ExecResult<()>;
type BoundedHandler =
    for<'store, 'module> fn(&mut Executor<'store, 'module>, usize, Instruction, u32) -> ExecResult<()>;

#[cold]
#[inline(never)]
fn instruction_handler_mismatch() -> ! {
    unreachable!("instruction handler mismatch")
}

macro_rules! define_unbudgeted_tail_dispatch {
    ($executor:ident, $instr_ptr:ident, $dispatch_next:ident, $dispatch_flow:ident;
     $($variant:ident $(($($arg:pat),*))? $({ $($field:ident),* })? => $body:expr),* $(,)?) => {
        #[inline(always)]
        fn handler_for(opcode: InstructionOpcode) -> UnbudgetedHandler {
            static HANDLERS: [UnbudgetedHandler; InstructionOpcode::COUNT] = {
                let mut handlers = [Unbudgeted::Unreachable as UnbudgetedHandler; InstructionOpcode::COUNT];
                $(handlers[InstructionOpcode::$variant as usize] = Unbudgeted::$variant;)*
                handlers
            };
            HANDLERS[opcode as usize]
        }

        $(
            #[allow(non_snake_case, unreachable_code, unused_imports, unused_macros, unused_variables)]
            fn $variant(
                $executor: &mut Executor<'_, '_>,
                instructions: &[Instruction],
                func_addr: FuncAddr,
                $instr_ptr: usize,
                instruction: Instruction,
            ) -> ExecResult<()> {
                macro_rules! $dispatch_next {
                    ($next_instr_ptr:expr) => {{
                        let next_instr_ptr = $next_instr_ptr;
                        let instruction = instructions[next_instr_ptr];
                        let handler = Self::handler_for(instruction.opcode());
                        become handler($executor, instructions, func_addr, next_instr_ptr, instruction);
                    }};
                }
                macro_rules! $dispatch_flow {
                    ($flow:expr) => {{
                        match $flow.next_instr_ptr() {
                            Some(next_instr_ptr) => {
                                if $executor.cf.func_addr != func_addr {
                                    $executor.cf.instr_ptr = next_instr_ptr;
                                    return Ok(());
                                }
                                $dispatch_next!(next_instr_ptr)
                            },
                            None => return cold!({
                                if !$executor.left {
                                    $executor.completed = true;
                                }
                                Ok(())
                            }),
                        }
                    }};
                }
                use tinywasm_types::Instruction::*;
                $(let $variant($($arg),*) = &instruction else {
                    cold!(instruction_handler_mismatch())
                };)?
                $(let $variant { $($field),* } = &instruction else {
                    cold!(instruction_handler_mismatch())
                };)?
                $body;
                $dispatch_next!($instr_ptr + 1)
            }
        )*
    };
}

macro_rules! define_bounded_tail_dispatch {
    ($executor:ident, $instr_ptr:ident, $dispatch_next:ident, $dispatch_flow:ident;
     $($variant:ident $(($($arg:pat),*))? $({ $($field:ident),* })? => $body:expr),* $(,)?) => {
        #[inline(always)]
        fn handler_for(opcode: InstructionOpcode) -> BoundedHandler {
            static HANDLERS: [BoundedHandler; InstructionOpcode::COUNT] = {
                let mut handlers = [Bounded::Unreachable as BoundedHandler; InstructionOpcode::COUNT];
                $(handlers[InstructionOpcode::$variant as usize] = Bounded::$variant;)*
                handlers
            };
            HANDLERS[opcode as usize]
        }

        $(
            #[allow(non_snake_case, unreachable_code, unused_imports, unused_macros, unused_variables)]
            fn $variant(
                $executor: &mut Executor<'_, '_>,
                $instr_ptr: usize,
                instruction: Instruction,
                instructions_until_checkpoint: u32,
            ) -> ExecResult<()> {
                macro_rules! $dispatch_next {
                    ($next_instr_ptr:expr) => {{
                        let next_instr_ptr = $next_instr_ptr;
                        if instructions_until_checkpoint == 0 {
                            return cold!({
                                $executor.cf.instr_ptr = next_instr_ptr;
                                Ok(())
                            });
                        }

                        let instruction = $executor.func.instructions[next_instr_ptr];
                        let handler = Self::handler_for(instruction.opcode());
                        become handler($executor, next_instr_ptr, instruction, instructions_until_checkpoint - 1);
                    }};
                }
                macro_rules! $dispatch_flow {
                    ($flow:expr) => {{
                        match $flow.next_instr_ptr() {
                            Some(next_instr_ptr) => $dispatch_next!(next_instr_ptr),
                            None => return cold!({
                                if $executor.left {
                                    $executor.chunk_left = instructions_until_checkpoint;
                                } else {
                                    $executor.completed = true;
                                }
                                Ok(())
                            }),
                        }
                    }};
                }
                use tinywasm_types::Instruction::*;
                $(let $variant($($arg),*) = &instruction else {
                    cold!(instruction_handler_mismatch())
                };)?
                $(let $variant { $($field),* } = &instruction else {
                    cold!(instruction_handler_mismatch())
                };)?
                $body;
                $dispatch_next!($instr_ptr + 1)
            }
        )*
    };
}

impl Unbudgeted {
    instruction_handlers!(define_unbudgeted_tail_dispatch);
}

impl Bounded {
    instruction_handlers!(define_bounded_tail_dispatch);

    /// Runs up to `chunk_left` (at least 1) instructions from `executor.cf`.
    #[inline(always)]
    fn run(executor: &mut Executor<'_, '_>, chunk_left: u32) -> ExecResult<()> {
        let instr_ptr = executor.cf.instr_ptr;
        let instruction = executor.func.instructions[instr_ptr];
        let handler = Self::handler_for(instruction.opcode());
        handler(executor, instr_ptr, instruction, chunk_left - 1)
    }
}

impl Executor<'_, '_> {
    /// Runs until the call completes (`None`) or continues in another module instance's frame.
    #[inline(always)]
    pub(crate) fn run_to_completion(mut self) -> Result<Option<CallFrame>> {
        loop {
            let func = self.func;
            let instructions = &func.instructions;
            let func_addr = self.cf.func_addr;
            let instr_ptr = self.cf.instr_ptr;
            let instruction = instructions[instr_ptr];
            let handler = Unbudgeted::handler_for(instruction.opcode());
            handler(&mut self, instructions, func_addr, instr_ptr, instruction)?;
            if self.completed || self.left {
                return Ok(self.left());
            }
        }
    }

    /// Runs `chunk_left` instructions, then checkpoint by checkpoint until `time_budget` has
    /// elapsed since `start`.
    #[cfg(feature = "std")]
    #[inline(always)]
    pub(crate) fn run_with_time_budget(
        mut self,
        start: crate::std::time::Instant,
        time_budget: core::time::Duration,
        mut chunk_left: u32,
    ) -> Result<RunEnd> {
        loop {
            if chunk_left != 0 {
                Bounded::run(&mut self, chunk_left)?;
                if let Some(end) = self.run_end() {
                    return Ok(end);
                }
            }
            chunk_left = CHECKPOINT_INTERVAL;
            if start.elapsed() >= time_budget {
                return cold!(Ok(RunEnd::State(ExecState::Suspended(self.cf))));
            }
        }
    }

    /// Runs `chunk_left` instructions, then checkpoint by checkpoint until the store's fuel is out.
    #[inline(always)]
    pub(crate) fn run_with_fuel(mut self, mut chunk_left: u32) -> Result<RunEnd> {
        self.fuel_metered = true;
        loop {
            if chunk_left != 0 {
                Bounded::run(&mut self, chunk_left)?;
                if let Some(end) = self.run_end() {
                    return Ok(end);
                }
            }
            chunk_left = CHECKPOINT_INTERVAL;
            self.store.execution_fuel = self.store.execution_fuel.saturating_sub(CHECKPOINT_INTERVAL);
            if self.store.execution_fuel == 0 {
                return cold!(Ok(RunEnd::State(ExecState::Suspended(self.cf))));
            }
        }
    }

    /// How a bounded chain that stopped ended, unless it stopped at a checkpoint.
    #[inline(always)]
    fn run_end(&self) -> Option<RunEnd> {
        if self.completed {
            return cold!(Some(RunEnd::State(ExecState::Completed)));
        }
        self.left().map(|frame| RunEnd::Left(frame, self.chunk_left))
    }
}
