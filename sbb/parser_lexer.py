import ply.lex as lex
import ply.yacc as yacc
import sys

# --- Lexer ---

tokens = (
    'ID',       # Function names (@main), Variable names (&fmt)
    'INT_LITERAL',
    'STR_LITERAL',
    'TYPE_INT',
    'TYPE_BYTE',
    'KW_DATA',
    'KW_SET',
    'KW_CALL',
    'KW_RET',
    'KW_PUSH',
    'KW_ADD',   # New instruction
    'KW_SUB',   # New arithmetic operation
    'KW_MUL',   # New arithmetic operation
    'KW_DIV',   # New arithmetic operation
    'KW_MOD',   # New arithmetic operation
    'KW_SHR',   # New bitwise operation
    'KW_SHL',   # New bitwise operation
    'KW_SWAP',  # New instruction
    'KW_POP',   # New instruction
    'KW_POPT',  # New instruction for pop-to-memory
    'KW_REMT',  # New instruction for remove-from-memory
    'KW_CMP',   # New instruction for comparison
    'KW_JMP_T', # New instruction for jump-if-true
    'KW_JMP_F', # New instruction for jump-if-false
    'KW_JMP',   # New instruction for unconditional jump
    'KW_DUP',   # New instruction for duplicate
    'KW_OVER',  # New instruction for over
    'KW_INC',   # New instruction for increment
    'KW_DEC',   # New instruction for decrement
    'VAR',      # New token for variable declaration
    'DOLLAR',   # New token for variable reference
    'CMP_OP',   # Comparison operators (equal, notequal, etc.)
    'LABEL',    # Label identifier (%label)
    'COLON',    # Colon symbol for label definitions
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
    r'[@&]?[a-zA-Z_][a-zA-Z_0-9]*' # Allows @name, &name, and name
    # Check for keywords (simple way, ply handles reserved words better)
    reserved = {
        'data': 'KW_DATA',
        'set': 'KW_SET',
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
    t.type = reserved.get(t.value, 'ID') # Check for reserved words
    return t

t_INT_LITERAL = r'\d+'
t_STR_LITERAL = r'"([^"\\]|\\.)*"' # Allows escaped quotes

# Symbols
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_LBRACE = r'\{'
t_RBRACE = r'\}'
t_LBRACKET = r'\['
t_RBRACKET = r'\]'
t_EQUALS = r'='
t_AMPERSAND = r'&' # Treat separately from &varname if needed, handled in ID for now
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
        self.value = value # String literal for byte array

class TypeSpec(Node):
    def __init__(self, base_type, size=None):
        self.base_type = base_type # 'byte' or 'int'
        self.size = size # Integer
        if base_type == 'int' and size is not None:
            print("Warning: size specification ignored for int type", file=sys.stderr)

class Param(Node): # New node for function parameters
    def __init__(self, type, name):
        self.type = type
        self.name = name

class FuncDecl(Node):
    def __init__(self, name, params, return_type, body):
        self.name = name
        self.params = params # List of Param nodes
        self.return_type = return_type # 'int'
        self.body = body # List of statements

class PushInst(Node):
    def __init__(self, value, value_type):
        self.value = value # INT_LITERAL (e.g., 10) or ID (e.g., &fmt, a)
        self.value_type = value_type # 'int' or 'id' or 'addr' etc.

class CallInst(Node):
    def __init__(self, func_name, arg_count):
        self.func_name = func_name # ID (@printf)
        self.arg_count = arg_count

class RetInst(Node):
    def __init__(self, type):
        self.type = type # 'int'

class AddInst(Node): # New instruction
    pass # No arguments needed

class SwapInst(Node): # New instruction
    pass # No arguments needed

class PopInst(Node): # New instruction
    pass # No arguments needed

class DupInst(Node): # New instruction for duplicate
    pass # No arguments needed

class OverInst(Node): # New instruction for over
    pass # No arguments needed

class IncInst(Node): # New instruction for increment
    pass # No arguments needed

class DecInst(Node): # New instruction for decrement
    pass # No arguments needed

class PoptInst(Node): # New instruction for pop-to-memory
    def __init__(self, dest):
        self.dest = dest # Destination address (e.g., &list_bytes)

class CmpInst(Node): # New instruction for comparison
    def __init__(self, op):
        self.op = op # Comparison operator ('equal', 'notequal', etc.)

class JmpInst(Node): # New instruction for jumps
    def __init__(self, type, label):
        self.type = type # 'jmp_t', 'jmp_f', or 'jmp'
        self.label = label # Label to jump to

class LabelNode(Node): # New node for label definitions
    def __init__(self, name):
        self.name = name # Label name without '%'

class RemtInst(Node): # New instruction for remove-from-memory
    def __init__(self, src):
        self.src = src # Source address (e.g., &list_bytes)

class VarDecl(Node): # Node for variable declaration
    def __init__(self, name):
        # Variable declaration takes value from stack
        self.name = name  # Variable name without $

class VarRef(Node): # New node for variable reference
    def __init__(self, name):
        self.name = name

class SubInst(Node): pass  # New arithmetic instruction
class MulInst(Node): pass  # New arithmetic instruction
class DivInst(Node): pass  # New arithmetic instruction
class ModInst(Node): pass  # New arithmetic instruction
class ShrInst(Node): pass  # New bitwise instruction
class ShlInst(Node): pass  # New bitwise instruction

# --- Parser (Grammar Rules) ---

# Precedence not strictly needed for this simple grammar yet
# precedence = (
# )

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
                   | func_declaration'''
    p[0] = p[1]

# Data Declaration: data &fmt = byte[14] "Hello, world!\n"
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
    elif len(p) == 5:  # byte[size]
        p[0] = TypeSpec('byte', int(p[3]))
    else:
        p[0] = None

# Function Declaration: set @sum(int a, int b) int { statements }
def p_func_declaration(p):
    'func_declaration : KW_SET ID LPAREN param_list_opt RPAREN TYPE_INT LBRACE statements RBRACE'
    p[0] = FuncDecl(p[2], p[4], p[6], p[8]) # Added params p[4], adjusted indices

def p_param_list_opt(p):
    '''param_list_opt : param_list
                      | empty'''
    # Check if p[1] exists and is not None (should correspond to param_list)
    if len(p) == 2 and p[1] is not None:
        p[0] = p[1]
    else: # Corresponds to empty or error
        p[0] = [] # Default to empty list

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
    p[0] = Param(TypeSpec(p[1]), p[2])  # Create TypeSpec wrapper for the type

def p_statements(p):
    '''statements : statements statement
                 | empty'''
    if len(p) == 3:
        p[0] = p[1] + [p[2]]
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
                 | var_declaration'''
    p[0] = p[1]

# Instructions
def p_push_instruction(p):
    '''push_instruction : KW_PUSH INT_LITERAL
                        | KW_PUSH ID
                        | KW_PUSH var_reference'''
    if isinstance(p[2], str) and p[2].isdigit():
        p[0] = PushInst(int(p[2]), 'int')
    elif isinstance(p[2], str):
        if p[2].startswith('&'):
            p[0] = PushInst(p[2], 'addr')
        else:
            p[0] = PushInst(p[2], 'id')
    elif isinstance(p[2], VarRef):
        p[0] = PushInst(p[2].name, 'var')
    else:
        print(f"Syntax error on push instruction: unexpected value {p[2]}", file=sys.stderr)
        p[0] = None

def p_call_instruction(p):
    'call_instruction : KW_CALL ID LPAREN INT_LITERAL RPAREN'
    p[0] = CallInst(p[2], int(p[4]))

def p_ret_instruction(p):
    'ret_instruction : KW_RET TYPE_INT'
    p[0] = RetInst(p[2])

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
    # Remove the '%' prefix from the label name
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
        print(f"Syntax error at '{p.value}' on line {p.lineno}", file=sys.stderr)
    else:
        print("Syntax error at EOF", file=sys.stderr)

# Helper to tokenize a string for debugging
def tokenize_string(s):
    lexer.input(s)
    tokens = []
    while True:
        tok = lexer.token()
        if not tok:
            break
        tokens.append(tok)
    return tokens

# Parse a string and return the AST
def parse_sbb(source_code):
    # Build the parser
    parser = yacc.yacc()
    
    # Reset lexer line counter
    lexer.lineno = 1
    
    try:
        # Parse the input
        ast = parser.parse(source_code, lexer=lexer)
        return ast
    except Exception as e:
        print(f"Error parsing SBB code: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None

# Debug helper to visualize the AST (can be improved)
def print_ast(node, indent=0):
    if node is None:
        return
    
    prefix = ' ' * indent
    
    if isinstance(node, list):
        for item in node:
            print_ast(item, indent)
    else:
        import json
        
        # Custom serializer for AST nodes
        def default_serializer(obj):
            if isinstance(obj, Node):
                return {f"{type(obj).__name__}": obj.__dict__}
            raise TypeError(f"Type {type(obj)} not serializable")
        
        # Print the node as JSON
        print(f"{prefix}{json.dumps(node, default=default_serializer, indent=2)}")

# --- Example Usage (for testing this module directly) ---
if __name__ == '__main__':
    test_code = """
    // this is a comment!
    data &fmt = byte[14] "Hello, world!\n"

    set @main() int {
        push &fmt
        call @printf
        ret int 0
    }
    """
    ast = parse_sbb(test_code)
    if ast:
        print("Parsing successful! AST:")
        # A simple way to visualize the AST
        print_ast(ast)
    else:
        print("Parsing failed.")
