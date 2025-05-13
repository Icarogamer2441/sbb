import ply.lex as lex
import ply.yacc as yacc
import sys

# --- Lexer ---

tokens = (
    'ID',       # Function names (@main), Variable names (&fmt), Macro names (puts)
    'INT_LITERAL',
    'STR_LITERAL',
    'TYPE_INT',
    'TYPE_BYTE',
    'KW_DATA',
    'KW_SET',
    'KW_MACRO',
    'KW_IMPORT', # New keyword for imports
    'KW_CALL',
    'KW_RET',
    'KW_PUSH',
    'KW_ADD',
    'KW_SUB',
    'KW_MUL',
    'KW_DIV',
    'KW_MOD',
    'KW_SHR',
    'KW_SHL',
    'KW_SWAP',
    'KW_POP',
    'KW_POPT',
    'KW_REMT',
    'KW_CMP',
    'KW_JMP_T',
    'KW_JMP_F',
    'KW_JMP',
    'KW_DUP',
    'KW_OVER',
    'KW_INC',
    'KW_DEC',
    'VAR',
    'DOLLAR',
    'CMP_OP',
    'LABEL',
    'COLON',
    'LPAREN', 'RPAREN',
    'LBRACE', 'RBRACE',
    'LBRACKET', 'RBRACKET',
    'EQUALS',
    'AMPERSAND',
    'COMMA',
)

# Ignore comments and whitespace
t_ignore_COMMENT = r'//.*'
t_ignore = ' \t\n'

def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

# Keywords
t_KW_DATA = r'data'
t_KW_SET = r'set'
t_KW_MACRO = r'macro'
t_KW_IMPORT = r'import' # Lexer rule for import keyword
t_KW_CALL = r'call'
t_KW_RET = r'ret'
t_KW_PUSH = r'push'
t_KW_ADD = r'add'
t_KW_SUB = r'sub'
t_KW_MUL = r'mul'
t_KW_DIV = r'div'
t_KW_MOD = r'mod'
t_KW_SHR = r'shr'
t_KW_SHL = r'shl'
t_KW_SWAP = r'swap'
t_KW_POP = r'pop'
t_KW_POPT = r'popt'
t_KW_REMT = r'remt'
t_KW_CMP = r'cmp'
t_KW_JMP_T = r'jmp_t'
t_KW_JMP_F = r'jmp_f'
t_KW_JMP = r'jmp'
t_KW_DUP = r'dup'
t_KW_OVER = r'over'
t_KW_INC = r'inc'
t_KW_DEC = r'dec'

# Comparison operators
def t_CMP_OP(t):
    r'equal|notequal|greater|less|greater_equal|less_equal'
    return t

# Types
t_TYPE_INT = r'int'
t_TYPE_BYTE = r'byte'

# Label definition
def t_LABEL(t):
    r'%[a-zA-Z_][a-zA-Z_0-9]*'
    return t

# Literals and Identifiers
def t_ID(t):
    r'[@&]?[a-zA-Z_][a-zA-Z_0-9]*' # Allows @name, &name, name, and macro_name
    reserved = {
        'data': 'KW_DATA',
        'set': 'KW_SET',
        'macro': 'KW_MACRO',
        'import': 'KW_IMPORT', # Added import to reserved words
        'call': 'KW_CALL',
        'ret': 'KW_RET',
        'push': 'KW_PUSH',
        'add': 'KW_ADD',
        'swap': 'KW_SWAP',
        'pop': 'KW_POP',
        'popt': 'KW_POPT',
        'remt': 'KW_REMT',
        'cmp': 'KW_CMP',
        'jmp_t': 'KW_JMP_T',
        'jmp_f': 'KW_JMP_F',
        'jmp': 'KW_JMP',
        'dup': 'KW_DUP',
        'over': 'KW_OVER',
        'inc': 'KW_INC',
        'dec': 'KW_DEC',
        'int': 'TYPE_INT',
        'byte': 'TYPE_BYTE',
        'var': 'VAR',
        '\$': 'DOLLAR',
        'sub': 'KW_SUB',
        'mul': 'KW_MUL',
        'div': 'KW_DIV',
        'mod': 'KW_MOD',
        'shr': 'KW_SHR',
        'shl': 'KW_SHL'
    }
    t.type = reserved.get(t.value, 'ID')
    return t

t_INT_LITERAL = r'\d+'
t_STR_LITERAL = r'"([^"\\]|\\.)*"'

# Symbols
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_LBRACE = r'\{'
t_RBRACE = r'\}'
t_LBRACKET = r'\['
t_RBRACKET = r'\]'
t_EQUALS = r'='
t_AMPERSAND = r'&'
t_COMMA = r','
t_COLON = r':'

t_VAR = r'var'
t_DOLLAR = r'\$'

def t_error(t):
    print(f"Illegal character '{t.value[0]}' at line {t.lexer.lineno}", file=sys.stderr)
    t.lexer.skip(1)

lexer = lex.lex()

# --- Parser (AST Node Definitions) ---

class Node:
    pass

class Program(Node):
    def __init__(self, declarations):
        self.declarations = declarations

class DataDecl(Node):
    def __init__(self, name, type_spec, value):
        self.name = name
        self.type_spec = type_spec
        self.value = value

class TypeSpec(Node):
    def __init__(self, base_type, size=None):
        self.base_type = base_type
        self.size = size
        if base_type == 'int' and size is not None:
            print("Warning: size specification ignored for int type", file=sys.stderr)

class Param(Node):
    def __init__(self, type, name):
        self.type = type
        self.name = name

class FuncDecl(Node):
    def __init__(self, name, params, return_type, body):
        self.name = name
        self.params = params
        self.return_type = return_type
        self.body = body

class MacroDef(Node): # AST Node for Macro Definition
    def __init__(self, name, body):
        self.name = name
        self.body = body # List of statement nodes

class MacroCall(Node): # AST Node for Macro Invocation (before expansion)
    def __init__(self, name):
        self.name = name

class ImportDecl(Node): # AST Node for Import Declaration
    def __init__(self, filepath):
        self.filepath = filepath # String literal

class PushInst(Node):
    def __init__(self, value, value_type):
        self.value = value
        self.value_type = value_type

class CallInst(Node):
    def __init__(self, func_name, arg_count):
        self.func_name = func_name
        self.arg_count = arg_count

class RetInst(Node):
    def __init__(self, type):
        self.type = type

class AddInst(Node): pass
class SwapInst(Node): pass
class PopInst(Node): pass
class DupInst(Node): pass
class OverInst(Node): pass
class IncInst(Node): pass
class DecInst(Node): pass

class PoptInst(Node):
    def __init__(self, dest):
        self.dest = dest

class CmpInst(Node):
    def __init__(self, op):
        self.op = op

class JmpInst(Node):
    def __init__(self, type, label):
        self.type = type
        self.label = label

class LabelNode(Node):
    def __init__(self, name):
        self.name = name

class RemtInst(Node):
    def __init__(self, src):
        self.src = src

class VarDecl(Node):
    def __init__(self, name):
        self.name = name

class VarRef(Node):
    def __init__(self, name):
        self.name = name

class SubInst(Node): pass
class MulInst(Node): pass
class DivInst(Node): pass
class ModInst(Node): pass
class ShrInst(Node): pass
class ShlInst(Node): pass

# --- Parser (Grammar Rules) ---

start = 'program'

def p_program(p):
    'program : declarations'
    p[0] = Program(p[1])

def p_declarations(p):
    '''declarations : declarations declaration
                  | declaration'''
    if len(p) == 3:
        p[0] = p[1] + [p[2]]
    else:
        p[0] = [p[1]]

def p_declaration(p):
    '''declaration : data_declaration
                   | func_declaration
                   | macro_definition
                   | import_declaration''' # Added import_declaration
    p[0] = p[1]

def p_import_declaration(p): # Grammar rule for import declaration
    'import_declaration : KW_IMPORT STR_LITERAL'
    p[0] = ImportDecl(p[2][1:-1]) # Store filepath without quotes

def p_data_declaration(p):
    'data_declaration : KW_DATA ID EQUALS type_spec STR_LITERAL'
    p[0] = DataDecl(p[2], p[4], p[5])

def p_type_spec(p):
    """type_spec : TYPE_BYTE
               | TYPE_BYTE LBRACKET INT_LITERAL RBRACKET
               | TYPE_INT"""
    if len(p) == 2 and p[1] == 'int':
        p[0] = TypeSpec('int')
    elif len(p) == 2 and p[1] == 'byte':
        p[0] = TypeSpec('byte')
    elif len(p) == 5:
        p[0] = TypeSpec('byte', int(p[3]))
    else:
        p[0] = None

def p_func_declaration(p):
    'func_declaration : KW_SET ID LPAREN param_list_opt RPAREN TYPE_INT LBRACE statements RBRACE'
    p[0] = FuncDecl(p[2], p[4], p[6], p[8])

def p_macro_definition(p): # Grammar rule for macro definition
    'macro_definition : KW_MACRO ID LBRACE statements RBRACE'
    p[0] = MacroDef(p[2], p[4]) # p[4] is the list of statements from the 'statements' rule

def p_param_list_opt(p):
    '''param_list_opt : param_list
                      | empty'''
    if len(p) == 2 and p[1] is not None:
        p[0] = p[1]
    else:
        p[0] = []

def p_param_list(p):
    '''param_list : param_list COMMA param
                 | param'''
    if len(p) == 4:
        p[0] = p[1] + [p[3]]
    else:
        p[0] = [p[1]]

def p_param(p):
    '''param : TYPE_INT ID
             | TYPE_BYTE ID'''
    p[0] = Param(TypeSpec(p[1]), p[2])

def p_statements(p):
    '''statements : statements statement
                 | empty'''
    if len(p) == 3:
        # p[2] could be a single statement or a list (from macro expansion, handled later)
        # For now, assume p[2] is a single statement node or None if error
        if p[2] is not None:
             p[0] = p[1] + [p[2]]
        else: # Error in statement, propagate existing statements
             p[0] = p[1]
    else:
        p[0] = []


def p_statement(p):
    '''statement : push_instruction
                 | call_instruction
                 | ret_instruction
                 | add_instruction
                 | sub_instruction
                 | mul_instruction
                 | div_instruction
                 | mod_instruction
                 | shr_instruction
                 | shl_instruction
                 | swap_instruction
                 | pop_instruction
                 | dup_instruction
                 | over_instruction
                 | inc_instruction
                 | dec_instruction
                 | popt_instruction
                 | remt_instruction
                 | cmp_instruction
                 | jmp_instruction
                 | label_definition
                 | var_declaration
                 | macro_invocation''' # Added macro_invocation
    p[0] = p[1]

def p_macro_invocation(p): # Grammar rule for macro invocation
    'macro_invocation : ID'
    # This ID is treated as a macro call. Validation happens in the expansion pass.
    p[0] = MacroCall(p[1])


def p_push_instruction(p):
    '''push_instruction : KW_PUSH INT_LITERAL
                        | KW_PUSH ID
                        | KW_PUSH var_reference'''
    if isinstance(p[2], str) and p[2].isdigit():
        p[0] = PushInst(int(p[2]), 'int')
    elif isinstance(p[2], str):
        if p[2].startswith('&'):
            p[0] = PushInst(p[2], 'addr')
        else: # Could be a parameter name or a global (currently unhandled directly here)
            p[0] = PushInst(p[2], 'id')
    elif isinstance(p[2], VarRef):
        p[0] = PushInst(p[2].name, 'var')
    else:
        print(f"Syntax error on push instruction: unexpected value {p[2]}", file=sys.stderr)
        p[0] = None # Error

def p_call_instruction(p):
    'call_instruction : KW_CALL ID LPAREN INT_LITERAL RPAREN'
    p[0] = CallInst(p[2], int(p[4]))

def p_ret_instruction(p):
    'ret_instruction : KW_RET TYPE_INT' # Assuming only int return for now
    p[0] = RetInst(p[2]) # Store 'int' or 'byte'

def p_add_instruction(p):
    'add_instruction : KW_ADD'
    p[0] = AddInst()

def p_swap_instruction(p):
    'swap_instruction : KW_SWAP'
    p[0] = SwapInst()

def p_pop_instruction(p):
    'pop_instruction : KW_POP'
    p[0] = PopInst()

def p_dup_instruction(p):
    'dup_instruction : KW_DUP'
    p[0] = DupInst()

def p_over_instruction(p):
    'over_instruction : KW_OVER'
    p[0] = OverInst()

def p_inc_instruction(p):
    'inc_instruction : KW_INC'
    p[0] = IncInst()

def p_dec_instruction(p):
    'dec_instruction : KW_DEC'
    p[0] = DecInst()

def p_popt_instruction(p):
    'popt_instruction : KW_POPT ID'
    p[0] = PoptInst(p[2])

def p_remt_instruction(p):
    'remt_instruction : KW_REMT ID'
    p[0] = RemtInst(p[2])

def p_cmp_instruction(p):
    'cmp_instruction : KW_CMP CMP_OP'
    p[0] = CmpInst(p[2])

def p_jmp_instruction(p):
    '''jmp_instruction : KW_JMP_T LABEL
                      | KW_JMP_F LABEL
                      | KW_JMP LABEL'''
    if p[1] == 'jmp_t':
        p[0] = JmpInst('jmp_t', p[2])
    elif p[1] == 'jmp_f':
        p[0] = JmpInst('jmp_f', p[2])
    elif p[1] == 'jmp':
        p[0] = JmpInst('jmp', p[2])

def p_label_definition(p):
    'label_definition : LABEL COLON'
    p[0] = LabelNode(p[1][1:])

def p_var_declaration(p):
    'var_declaration : VAR DOLLAR ID'
    p[0] = VarDecl(p[3])

def p_var_reference(p):
    'var_reference : DOLLAR ID'
    p[0] = VarRef(p[2])

def p_empty(p):
    'empty :'
    pass

def p_sub_instruction(p):
    'sub_instruction : KW_SUB'
    p[0] = SubInst()

def p_mul_instruction(p):
    'mul_instruction : KW_MUL'
    p[0] = MulInst()

def p_div_instruction(p):
    'div_instruction : KW_DIV'
    p[0] = DivInst()

def p_mod_instruction(p):
    'mod_instruction : KW_MOD'
    p[0] = ModInst()

def p_shr_instruction(p):
    'shr_instruction : KW_SHR'
    p[0] = ShrInst()

def p_shl_instruction(p):
    'shl_instruction : KW_SHL'
    p[0] = ShlInst()

def p_error(p):
    if p:
        print(f"Syntax error at '{p.value}' on line {p.lineno(0)}", file=sys.stderr)
    else:
        print("Syntax error at EOF", file=sys.stderr)

# --- Macro Expansion ---

def expand_macros_in_statements(statements, defined_macros):
    expanded_stmts = []
    for stmt in statements:
        if isinstance(stmt, MacroCall):
            if stmt.name in defined_macros:
                # Recursively expand macros within the macro's body
                # This handles nested macro calls.
                # Important: Deepcopy or ensure AST nodes are not shared if mutable
                # For this structure, re-processing the list of nodes should be fine.
                macro_body_stmts = defined_macros[stmt.name]
                expanded_macro_body = expand_macros_in_statements(list(macro_body_stmts), defined_macros) # Pass a copy
                expanded_stmts.extend(expanded_macro_body)
            else:
                # This error should ideally be caught by the parser if macro names are restricted
                # or by a semantic analysis phase. Raising here if it slips through.
                raise ValueError(f"Error: Call to undefined macro '{stmt.name}'.")
        elif isinstance(stmt, FuncDecl): # Should not happen if FuncDecl is top-level
             stmt.body = expand_macros_in_statements(stmt.body, defined_macros)
             expanded_stmts.append(stmt)
        # Add handling for other complex statements if they can contain statements (e.g., if/loops)
        # else if isinstance(stmt, IfNode):
        #    stmt.true_branch = expand_macros_in_statements(stmt.true_branch, defined_macros)
        #    stmt.false_branch = expand_macros_in_statements(stmt.false_branch, defined_macros)
        #    expanded_stmts.append(stmt)
        else:
            expanded_stmts.append(stmt)
    return expanded_stmts

def expand_macros_in_ast(node, defined_macros_map):
    """
    Traverses the AST, collects macro definitions, and expands macro calls.
    Modifies the AST in place.
    `defined_macros_map` is an empty dictionary passed in, filled by this function.
    """
    if not isinstance(node, Program):
        # Should only be called on the root Program node
        return node

    # Pass 1: Collect macro definitions
    # Store them in defined_macros_map and prepare a list of non-macro declarations
    new_program_declarations = []
    for decl in node.declarations:
        if isinstance(decl, MacroDef):
            if decl.name in defined_macros_map:
                # Allowing redefinition, last one wins (or could be an error)
                print(f"Warning: Macro '{decl.name}' redefined. Using the new definition.", file=sys.stderr)
            defined_macros_map[decl.name] = decl.body # Store the list of statement nodes
        else:
            new_program_declarations.append(decl)

    # Pass 2: Expand macros in function bodies using the collected macros
    # Iterate over the new_program_declarations which now only contains FuncDecl and DataDecl
    for i, decl in enumerate(new_program_declarations):
        if isinstance(decl, FuncDecl):
            # Create a fresh copy of the body for expansion to avoid issues with shared lists if FuncDecl was copied
            decl.body = expand_macros_in_statements(list(decl.body), defined_macros_map)
            new_program_declarations[i] = decl # Update the declaration in the list

    # Update the program's declarations to the list without MacroDefs and with expanded macros
    node.declarations = new_program_declarations
    return node


# Helper to tokenize a string for debugging
def tokenize_string(s):
    lexer.input(s)
    tokens_list = []
    while True:
        tok = lexer.token()
        if not tok:
            break
        tokens_list.append(tok)
    return tokens_list

# Parse a string and return the AST
def parse_sbb(source_code):
    parser = yacc.yacc()
    # parser.macros = {} # No longer needed here if expansion is a separate pass
    lexer.lineno = 1
    try:
        ast = parser.parse(source_code, lexer=lexer)
        return ast
    except Exception as e:
        print(f"Error parsing SBB code: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None

# Debug helper to visualize the AST
def print_ast(node, indent=0):
    if node is None:
        return
    prefix = ' ' * indent
    if isinstance(node, list):
        for item in node:
            print_ast(item, indent)
    else:
        # A more robust way to print, handling node attributes
        node_type_name = type(node).__name__
        attrs = {}
        if hasattr(node, '__dict__'):
            for k, v in node.__dict__.items():
                if isinstance(v, list) and all(isinstance(i, Node) for i in v):
                    # Placeholder for lists of nodes to avoid deep recursion in simple print
                    attrs[k] = f"[List of {len(v)} Node(s)]"
                elif isinstance(v, Node):
                    attrs[k] = type(v).__name__
                else:
                    attrs[k] = v
            print(f"{prefix}{node_type_name}: {attrs}")
            # Recursively print child nodes if they are lists of nodes
            for k, v in node.__dict__.items():
                if isinstance(v, list) and all(isinstance(i, Node) for i in v):
                    print(f"{prefix}  {k}:")
                    for item in v:
                        print_ast(item, indent + 4)
                elif isinstance(v, Node) and k != 'type_spec' and k != 'type': # Avoid simple type nodes unless complex
                     print(f"{prefix}  {k}:")
                     print_ast(v, indent + 4)

        else: # For simple nodes or literals not having __dict__
            print(f"{prefix}{node_type_name}: {node}")


if __name__ == '__main__':
    test_code_macros = """
    data &hello_msg = byte[13] "Hello Macro!"
    data &newline = byte[2] "\n"
    data &fmt_str = byte[3] "%s"

    macro print_str { // Macro to print a string assumed to be on stack
        push &fmt_str  // Format string
        swap           // Swap with the string to print
        push &newline  // Newline string
        swap           // Swap with format string (now: string, newline, fmt_str)
                       // This order is not right for printf. Let's simplify.
                       // Assume printf takes (fmt, arg1, arg2...)
                       // For one string: push string_addr; push fmt_addr; call printf(2)
        // Corrected logic for puts-like macro:
        // Assumes string to print is already on stack.
        // 1. push format_string_for_string_and_newline ("%s\n")
        // 2. swap (to get string_to_print, format_string)
        // 3. call printf(2)

        // Simpler macro for the example:
        // This macro expects the string to print to be on the stack.
        // It then pushes the format string "%s\n" and calls printf.
    }
    
    // Let's use the example's macro structure
    data &str_fmt_example = byte[3] "%s\n" // from examples/macros.sbb

    macro puts_example {
        push &str_fmt_example // Pushes "%s\n"
        swap                  // Swaps with the string already on stack
        call @printf(2)       // Calls printf
    }

    set @main() int {
        push &hello_msg
        puts_example      // Call the macro

        push 0
        ret int
    }
    """
    print("--- Tokenizing Macro Test Code ---")
    tokens_found = tokenize_string(test_code_macros)
    for tk in tokens_found:
        print(tk)
    print("--- Parsing Macro Test Code ---")
    ast = parse_sbb(test_code_macros)
    if ast:
        print("\n--- Initial AST (Before Macro Expansion) ---")
        print_ast(ast)

        # Expand Macros
        print("\n--- Expanding Macros ---")
        macros_map = {}
        expanded_ast = expand_macros_in_ast(ast, macros_map) # ast is modified in-place
        print("Macros collected:", {name: type(body) for name, body in macros_map.items()})


        print("\n--- Final AST (After Macro Expansion) ---")
        print_ast(expanded_ast) # Print the modified AST
    else:
        print("Parsing failed for macro test code.")
