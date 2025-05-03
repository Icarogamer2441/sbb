from .parser_lexer import (
    Node, Program, DataDecl, FuncDecl, Param, PushInst, CallInst, RetInst,
    AddInst, SubInst, MulInst, DivInst, ModInst, ShrInst, ShlInst,  # Arithmetic ops
    SwapInst, PopInst, PoptInst, RemtInst, CmpInst, JmpInst, LabelNode, DupInst, OverInst,
    VarDecl, VarRef, IncInst, DecInst # Import new nodes
)
import sys

# Basic x86_64 GAS code generator

class CodeGenerator:
    def __init__(self):
        self.assembly_code = []
        self.data_section = []
        self.text_section = []
        self.label_count = 0
        self.current_func_params = {} # To map param names to stack locations [%rbp+offset]
        self.current_func_vars = {}  # Track local variables
        # self.param_reg_map = ['%rdi', '%rsi', '%rdx', '%rcx', '%r8', '%r9'] # SysV ABI - No longer used for internal mapping

    def generate_label(self, prefix='L'):
        self.label_count += 1
        return f"{prefix}{self.label_count}"

    def add_instr(self, instr, indent=True):
        self.text_section.append(('\t' if indent else '') + instr)

    def visit(self, node):
        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise NotImplementedError(f"No visit_{type(node).__name__} method")

    def visit_Program(self, node):
        # Process data declarations first
        for decl in node.declarations:
            if isinstance(decl, DataDecl):
                self.visit(decl)

        self.text_section.append(".text")
        # Process function declarations next
        for decl in node.declarations:
            if isinstance(decl, FuncDecl):
                self.visit(decl)

        # Combine sections
        self.assembly_code.append(".data")
        self.assembly_code.extend(self.data_section)
        self.assembly_code.append("") # Newline separator
        self.assembly_code.extend(self.text_section)

        return '\n'.join(self.assembly_code) + '\n' # Ensure trailing newline

    def visit_DataDecl(self, node):
        # Handle both byte and int data declarations
        label = node.name.lstrip('&')  # Remove leading '&' for label
        
        if node.type_spec.base_type == 'byte':
            # Byte array (string) handling
            string_value = node.value[1:-1]  # Remove quotes
            # Escape special characters (e.g., newlines, quotes)
            string_value = string_value.replace('\\n', '\\n').replace('\\"', '\\"')
            # Ensure null termination for C compatibility
            if not string_value.endswith('\\0'):
                string_value += '\\0'
            self.data_section.append(f"{label}:\t.string \"{string_value}\"")
        elif node.type_spec.base_type == 'int':
            # Integer value handling
            int_value = node.value
            self.data_section.append(f"{label}:\t.quad {int_value}")
        else:
            print(f"Error: Unknown data type {node.type_spec.base_type}", file=sys.stderr)

    def visit_FuncDecl(self, node):
        self.current_func_vars = {}  # Reset variable tracking
        label = node.name.lstrip('@')
        
        # Map parameter names to stack locations relative to %rbp
        # For byte parameters, we'll need to handle them differently (only use lower byte)
        self.current_func_params = {}
        for i, param in enumerate(node.params):
            offset = 16 + i * 8
            if param.type.base_type == 'byte':
                self.current_func_params[param.name] = {
                    'location': f"{offset}(%rbp)",
                    'type': 'byte'
                }
            else:  # int
                self.current_func_params[param.name] = {
                    'location': f"{offset}(%rbp)",
                    'type': 'int'
                }

        # print(f"[Codegen] Visiting FuncDecl for {node.name} with param map: {self.current_func_params}") # Debug
        self.add_instr(f".globl {label}", indent=False)
        self.add_instr(f"{label}:", indent=False)

        self.add_instr("pushq %rbp")
        self.add_instr("movq %rsp, %rbp")
        
        # Reserve space for local variables
        # We'll allocate this at the beginning to ensure %rbp offsets stay consistent
        # Count statements to estimate how many locals we need
        # This is a simplification - ideally we'd analyze the AST to get an exact count
        estimated_vars = sum(1 for stmt in node.body if isinstance(stmt, VarDecl))
        if estimated_vars > 0:
            self.add_instr(f"subq ${estimated_vars * 8}, %rsp")  # Reserve space

        # Visit statements
        for stmt in node.body:
            self.visit(stmt)

        self.current_func_params = {} # Clear param map
        self.current_func_vars = {}  # Clear variable tracking

    def visit_PushInst(self, node):
        if node.value_type == 'var':
            # Handle variable references
            if node.value in self.current_func_vars:
                self.add_instr(f"# Push variable ${node.value}")
                self.add_instr(f"movq {self.current_func_vars[node.value]}, %rax")
                self.add_instr("pushq %rax")
            else:
                raise Exception(f"Unknown variable ${node.value}")
        elif node.value_type == 'int':
            self.add_instr(f"pushq ${node.value}")
        elif node.value_type == 'addr':
            label = node.value.lstrip('&')
            self.add_instr(f"leaq {label}(%rip), %rax")
            self.add_instr(f"pushq %rax")
        elif node.value_type == 'id':
            param_name = node.value
            if param_name in self.current_func_params:
                param_info = self.current_func_params[param_name]
                if param_info['type'] == 'byte':
                    # For byte parameters, zero-extend to 64 bits
                    self.add_instr(f"movzbq {param_info['location']}, %rax")
                else:  # int
                    self.add_instr(f"movq {param_info['location']}, %rax")
                self.add_instr(f"pushq %rax")
            else:
                print(f"Error: Unknown identifier '{param_name}' in push instruction.", file=sys.stderr)
                self.add_instr(f"# Error: push unknown id {param_name}")
        else:
            print(f"Error: Unknown push type '{node.value_type}'", file=sys.stderr)
            self.add_instr(f"# Error: unknown push type {node.value_type}")

    def visit_CallInst(self, node):
        func_name = node.func_name.lstrip('@')
        arg_count = node.arg_count
        is_external = func_name in ['printf'] # Simple heuristic
        suffix = "@PLT" if is_external else ""

        if is_external:
            # External Call (printf): Pop args from our stack into SysV ABI registers
            regs_for_args = ['%rdi', '%rsi', '%rdx', '%rcx', '%r8', '%r9']
            if arg_count > len(regs_for_args):
                print(f"Error: Too many arguments ({arg_count}) for external call {func_name}. Max {len(regs_for_args)} supported.", file=sys.stderr)
                # Handle error appropriately
                return

            self.add_instr(f"# Loading {arg_count} args into registers for external call {func_name}")
            # Pop arguments in reverse order into registers
            for i in range(arg_count - 1, -1, -1):
                self.add_instr(f"popq {regs_for_args[i]}")

            # Stack alignment preamble
            self.add_instr("movq %rsp, %rbx")
            self.add_instr("andq $-16, %rsp")
            self.add_instr("pushq %rbx")
            self.add_instr("subq $8, %rsp")
            self.add_instr("movl $0, %eax") # 0 vector args

            # Make the call
            self.add_instr(f"call {func_name}{suffix}")

            # Stack alignment postamble
            self.add_instr("addq $8, %rsp")
            self.add_instr("popq %rsp")

            # Result is typically in %rax, push it back onto our stack? Maybe not needed for printf.
            # self.add_instr("pushq %rax")
        else:
            # Internal Call (e.g. @sum)
            # Assume args were already pushed by caller according to our convention
            self.add_instr(f"# Calling internal function {func_name}")
            self.add_instr(f"call {func_name}{suffix}")

            # Caller cleans up arguments from the stack
            if arg_count > 0:
                bytes_to_clean = 8 * arg_count
                self.add_instr(f"addq ${bytes_to_clean}, %rsp")

            # Push the result (which is in %rax) back onto the stack
            self.add_instr("pushq %rax")

    def visit_RetInst(self, node):
        if node.type == 'byte':
            # For strings, return pointer in %rax
            self.add_instr("movq %rsp, %rax")
        else:  # int
            self.add_instr("popq %rax")
        self.add_instr("leave")
        self.add_instr("ret")

    def visit_AddInst(self, node):
        self.add_instr("popq %rbx")   # Pop second operand into rbx
        self.add_instr("popq %rax")   # Pop first operand into rax
        self.add_instr("addq %rbx, %rax") # Add rbx into rax
        self.add_instr("pushq %rax")  # Push result back onto stack

    def visit_SwapInst(self, node):
        self.add_instr("popq %rbx")   # Pop top into rbx
        self.add_instr("popq %rax")   # Pop next into rax
        self.add_instr("pushq %rbx")  # Push original top (rbx)
        self.add_instr("pushq %rax")  # Push original second (rax)

    def visit_PopInst(self, node):
        self.add_instr("addq $8, %rsp") # Effectively discard top by moving stack pointer

    def visit_DupInst(self, node):
        # Duplicate the top of the stack
        self.add_instr("# Duplicate top of stack")
        self.add_instr("movq (%rsp), %rax")  # Load top of stack into rax
        self.add_instr("pushq %rax")         # Push it back

    def visit_OverInst(self, node):
        # Copy second stack item to top (over)
        self.add_instr("# Copy second stack item to top (over)")
        self.add_instr("movq 8(%rsp), %rax")  # Load second item into rax
        self.add_instr("pushq %rax")          # Push it to top

    def visit_IncInst(self, node):
        # Increment the top of the stack
        self.add_instr("# Increment top of stack")
        self.add_instr("popq %rax")   # Pop top into rax
        self.add_instr("incq %rax")   # Increment rax
        self.add_instr("pushq %rax")  # Push result back onto stack

    def visit_DecInst(self, node):
        # Decrement the top of the stack
        self.add_instr("# Decrement top of stack")
        self.add_instr("popq %rax")   # Pop top into rax
        self.add_instr("decq %rax")   # Decrement rax
        self.add_instr("pushq %rax")  # Push result back onto stack

    def visit_RemtInst(self, node):
        # Push byte from memory onto stack (zero-extended to 64 bits)
        src = node.src.lstrip('&')
        self.add_instr(f"# Push byte from {src} to stack")
        self.add_instr(f"movzbq {src}(%rip), %rax")  # Load byte, zero extend to 64 bits
        self.add_instr("pushq %rax")

    def visit_PoptInst(self, node):
        # Generate unique labels for this instruction
        find_label = self.generate_label("find_null")
        found_label = self.generate_label("found_null")
        
        # Pop value and store byte to memory (with auto-increment)
        self.add_instr("# Pop and store byte to memory")
        self.add_instr("popq %rax")  # Pop value
        dest = node.dest.lstrip('&')
        # Find next null byte and store there
        self.add_instr(f"leaq {dest}(%rip), %rbx")
        self.add_instr("movq $0, %rcx")  # Counter
        self.add_instr(f"{find_label}:")
        self.add_instr("cmpb $0, (%rbx,%rcx,1)")
        self.add_instr(f"je {found_label}")
        self.add_instr("incq %rcx")
        self.add_instr(f"jmp {find_label}")
        self.add_instr(f"{found_label}:")
        self.add_instr("movb %al, (%rbx,%rcx,1)")  # Store at null position
        self.add_instr("incq %rcx")
        self.add_instr("movb $0, (%rbx,%rcx,1)")  # Add new null terminator

    def visit_CmpInst(self, node):
        # Comparison assumes two values on the stack: second value, first value (top)
        # We compare first value against second value (i.e., first == second)
        self.add_instr(f"# Comparison operator: {node.op}")
        self.add_instr("popq %rbx")  # Pop first operand into rbx (right-hand side of comparison)
        self.add_instr("popq %rax")  # Pop second operand into rax (left-hand side of comparison)
        self.add_instr("cmpq %rbx, %rax")  # Compare rax with rbx
        
        # For each comparison operator, we'll push 1 (true) or 0 (false) onto the stack
        result_label = self.generate_label("cmp_result")
        end_label = self.generate_label("cmp_end")
        
        if node.op == "equal":
            self.add_instr("je " + result_label)  # Jump if equal
        elif node.op == "notequal":
            self.add_instr("jne " + result_label)  # Jump if not equal
        elif node.op == "greater":
            self.add_instr("jg " + result_label)  # Jump if greater
        elif node.op == "less":
            self.add_instr("jl " + result_label)  # Jump if less
        elif node.op == "greater_equal":
            self.add_instr("jge " + result_label)  # Jump if greater or equal
        elif node.op == "less_equal":
            self.add_instr("jle " + result_label)  # Jump if less or equal
        else:
            print(f"Error: Unknown comparison operator '{node.op}'", file=sys.stderr)
            self.add_instr(f"# Error: Unknown comparison operator {node.op}")
            # Default to equality
            self.add_instr("je " + result_label)
        
        # False case: push 0
        self.add_instr("pushq $0")
        self.add_instr("jmp " + end_label)
        
        # True case: push 1
        self.add_instr(result_label + ":", indent=False)
        self.add_instr("pushq $1")
        
        # End of comparison
        self.add_instr(end_label + ":", indent=False)

    def visit_JmpInst(self, node):
        label_name = node.label.lstrip('%')  # Remove '%' prefix if present
        
        if node.type == "jmp_t":
            # Jump if true (non-zero)
            self.add_instr("# Jump if true")
            self.add_instr("popq %rax")  # Pop condition result into rax
            self.add_instr("testq %rax, %rax")  # Check if rax is non-zero
            self.add_instr("jnz " + label_name)  # Jump if not zero
        elif node.type == "jmp_f":
            # Jump if false (zero)
            self.add_instr("# Jump if false")
            self.add_instr("popq %rax")  # Pop condition result into rax
            self.add_instr("testq %rax, %rax")  # Check if rax is zero
            self.add_instr("jz " + label_name)  # Jump if zero
        elif node.type == "jmp":
            # Unconditional jump
            self.add_instr("# Unconditional jump")
            self.add_instr("jmp " + label_name)
        else:
            print(f"Error: Unknown jump type '{node.type}'", file=sys.stderr)
            self.add_instr(f"# Error: Unknown jump type {node.type}")

    def visit_LabelNode(self, node):
        # Simply output the label
        self.add_instr(node.name + ":", indent=False)

    def visit_VarDecl(self, node):
        # Pop value from stack and store it in the variable
        self.add_instr(f"# Allocate and initialize variable ${node.name}")
        # Track variable offset from RBP (negative for locals)
        var_offset = 8 * (len(self.current_func_vars) + 1)
        self.current_func_vars[node.name] = f"-{var_offset}(%rbp)"
        
        # Pop the value from stack and store it
        self.add_instr("popq %rax")  # Pop value into rax
        self.add_instr(f"movq %rax, -{var_offset}(%rbp)")  # Store in variable

    def visit_VarRef(self, node):
        # Push variable value onto stack
        if node.name in self.current_func_vars:
            self.add_instr(f"# Push variable ${node.name}")
            self.add_instr(f"movq {self.current_func_vars[node.name]}, %rax")
            self.add_instr("pushq %rax")
        else:
            raise Exception(f"Unknown variable ${node.name}")

    def visit_SubInst(self, node):
        self.add_instr("# Subtraction: second - first (second is deeper in stack)")
        self.add_instr("popq %rbx")  # First operand (top of stack)
        self.add_instr("popq %rax")  # Second operand
        # Note: in x86, this is rax = rax - rbx
        self.add_instr("subq %rbx, %rax")  # rax = rax - rbx
        self.add_instr("pushq %rax")  # Push result

    def visit_MulInst(self, node):
        self.add_instr("# Multiplication")
        self.add_instr("popq %rbx")
        self.add_instr("popq %rax")
        self.add_instr("imulq %rbx, %rax")
        self.add_instr("pushq %rax")

    def visit_DivInst(self, node):
        self.add_instr("# Division")
        self.add_instr("popq %rbx")
        self.add_instr("popq %rax")
        self.add_instr("cqto")  # Sign extend rax to rdx:rax
        self.add_instr("idivq %rbx")
        self.add_instr("pushq %rax")  # Quotient

    def visit_ModInst(self, node):
        self.add_instr("# Modulo")
        self.add_instr("popq %rbx")
        self.add_instr("popq %rax")
        self.add_instr("cqto")
        self.add_instr("idivq %rbx")
        self.add_instr("pushq %rdx")  # Remainder

    def visit_ShrInst(self, node):
        self.add_instr("# Shift right")
        self.add_instr("popq %rcx")  # Shift count
        self.add_instr("popq %rax")  # Value
        self.add_instr("shrq %cl, %rax")
        self.add_instr("pushq %rax")

    def visit_ShlInst(self, node):
        self.add_instr("# Shift left")
        self.add_instr("popq %rcx")  # Shift count
        self.add_instr("popq %rax")  # Value
        self.add_instr("shlq %cl, %rax")
        self.add_instr("pushq %rax")

def generate_assembly(ast):
    """Generates GAS x86_64 assembly from the AST."""
    if not ast:
        return "# Error: No AST provided to code generator\n"
    generator = CodeGenerator()
    assembly = generator.visit(ast)
    return assembly

# --- Example Usage (for testing this module directly) ---
if __name__ == '__main__':
    from .parser_lexer import parse_sbb
    test_code = """
    data &fmt = byte[14] "Hello, world!\n"

    set @main() int {
        push &fmt
        call @printf
        ret int 0
    }
    """
    ast = parse_sbb(test_code)
    if ast:
        print("AST generated, proceeding to code generation...")
        assembly_code = generate_assembly(ast)
        print("--- Generated Assembly ---")
        print(assembly_code)
        print("------------------------")
    else:
        print("Parsing failed, cannot generate code.")
