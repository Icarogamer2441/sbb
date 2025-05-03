"""
SBB Compiler Advanced Optimizer
This module contains various optimization passes to transform the SBB AST
into a more efficient form before code generation.
"""

from collections import defaultdict

def fold_constants(node):
    """Recursively performs constant folding on the AST."""
    if isinstance(node, tuple):
        node_type = node[0]

        # Recursively optimize children first
        optimized_children = [fold_constants(child) for child in node[1:]]

        if node_type == 'binop':
            op, left, right = optimized_children
            # Check if both operands are numbers after potential folding
            if isinstance(left, tuple) and left[0] == 'number' and \
               isinstance(right, tuple) and right[0] == 'number':
                left_val = left[1]
                right_val = right[1]
                if op == '+':
                    return ('number', left_val + right_val)
                elif op == '-':
                    return ('number', left_val - right_val)
                elif op == '*':
                    return ('number', left_val * right_val)
                elif op == '/':
                    # Avoid division by zero
                    if right_val == 0:
                         raise ZeroDivisionError("Compile-time division by zero detected")
                    # Use integer division consistent with typical stack machines
                    return ('number', left_val // right_val)
                elif op == '%':
                    if right_val == 0:
                         raise ZeroDivisionError("Compile-time modulo by zero detected")
                    return ('number', left_val % right_val)
                elif op == '<<':
                    if right_val < 0:
                        print("Warning: Negative shift amount in constant folding")
                        return ('binop', op, left, right)
                    return ('number', left_val << right_val)
                elif op == '>>':
                    if right_val < 0:
                        print("Warning: Negative shift amount in constant folding")
                        return ('binop', op, left, right)
                    return ('number', left_val >> right_val)
                elif op == '&':
                    return ('number', left_val & right_val)
                elif op == '|':
                    return ('number', left_val | right_val)
                elif op == '^':
                    return ('number', left_val ^ right_val)
            # If not foldable, return the binop with potentially optimized children
            return ('binop', op, left, right)

        elif node_type in ['program', 'statements', 'func_def', 'params', 'args', 'call']:
            # Reconstruct nodes with potentially optimized children
            return (node_type,) + tuple(optimized_children)
        
        elif node_type in ['push', 'number', 'identifier', 'param', 'arg', 'return']:
             # These nodes don't have children that need folding in this context, 
             # or their children are handled by the general recursion.
             # Return the node as is (or with optimized children if applicable, handled by the initial loop)
             if len(node) > 1:
                return (node_type,) + tuple(optimized_children)
             else:
                return node # Nodes like ('return',) 

        else:
             # Unknown node type, return as is
            return (node_type,) + tuple(optimized_children)

    # If it's not a tuple (e.g., a raw number or identifier in a list), return it directly
    return node

def eliminate_dead_code(node):
    """Recursively performs advanced dead code elimination on the AST."""
    if isinstance(node, tuple):
        node_type = node[0]

        # Special handling for function bodies and statement blocks
        if node_type == 'func_def':
            # ('func_def', name, params, statements)
            name = node[1]
            params = eliminate_dead_code(node[2])
            statements = node[3]
            
            if isinstance(statements, tuple) and statements[0] == 'statements':
                new_statements = []
                return_encountered = False
                has_conditional_jump = False
                
                # First pass: identify if there are conditional jumps
                for stmt in statements[1:]:
                    if isinstance(stmt, tuple) and stmt[0] == 'jmp' and stmt[1] in ['jmp_t', 'jmp_f']:
                        has_conditional_jump = True
                        break
                
                # If function has conditional jumps, we need to be more careful
                # as code after a return might still be reachable by jumps
                if has_conditional_jump:
                    labels_after_return = set()
                    current_labels = set()
                    
                    # Collect all labels that appear after returns
                    for i, stmt in enumerate(statements[1:]):
                        if isinstance(stmt, tuple) and stmt[0] == 'label':
                            current_labels.add(stmt[1])
                        elif stmt[0] == 'return':
                            for j in range(i+1, len(statements)-1):
                                next_stmt = statements[j+1]
                                if isinstance(next_stmt, tuple) and next_stmt[0] == 'label':
                                    labels_after_return.add(next_stmt[1])
                    
                    # Only keep statements up to last necessary label
                    for stmt in statements[1:]:
                        new_statements.append(eliminate_dead_code(stmt))
                        if (isinstance(stmt, tuple) and stmt[0] == 'label' and 
                            stmt[1] not in labels_after_return and 
                            all(lbl not in labels_after_return for lbl in current_labels)):
                            break
                else:
                    # Simpler case: no conditional jumps, just eliminate code after return
                    for stmt in statements[1:]:
                        if return_encountered:
                            # Only keep label definitions after return for possible jumps to them
                            if isinstance(stmt, tuple) and stmt[0] == 'label':
                                new_statements.append(eliminate_dead_code(stmt))
                        else:
                            new_statements.append(eliminate_dead_code(stmt))
                            if isinstance(stmt, tuple) and stmt[0] == 'return':
                                return_encountered = True
                
                optimized_statements = ('statements',) + tuple(new_statements)
                return ('func_def', name, params, optimized_statements)
            else:
                # Handle the case if statements is not as expected
                return ('func_def', name, params, eliminate_dead_code(statements))
        
        # If not a special node, recursively process all children
        optimized_children = [eliminate_dead_code(child) for child in node[1:]]
        return (node_type,) + tuple(optimized_children)

    # Not a tuple, return as is
    return node

def constant_propagation(node, var_values=None):
    """Propagates constant values of variables through the AST."""
    if var_values is None:
        var_values = {}  # Track known constant values of variables
    
    if isinstance(node, tuple):
        node_type = node[0]
        
        if node_type == 'func_def':
            # Create a new variable tracking context for this function
            local_var_values = var_values.copy()
            name = node[1]
            params = constant_propagation(node[2], local_var_values)
            
            # Process function body with local variable tracking
            statements = constant_propagation(node[3], local_var_values)
            return ('func_def', name, params, statements)
        
        elif node_type == 'var_decl':
            # Track variable declarations that assign constants
            var_name = node[1]
            
            # Check if the variable is being assigned a constant value
            # This requires finding the preceding push instruction
            # For simplicity, let's assume the AST has been transformed so that
            # constant values are directly available
            if len(node) > 2 and isinstance(node[2], tuple) and node[2][0] == 'number':
                var_values[var_name] = node[2][1]  # Store the constant value
            else:
                # If not a constant, remove any previous knowledge about this var
                if var_name in var_values:
                    del var_values[var_name]
            
            return node  # Return the original var_decl node
        
        elif node_type == 'push' and node[1] == 'var':
            # Replace variable references with constants if known
            var_name = node[2]
            if var_name in var_values:
                # Replace with direct value push
                return ('push', 'int', var_values[var_name])
            return node
        
        # If not a special node, recursively process all children
        optimized_children = [constant_propagation(child, var_values) for child in node[1:]]
        return (node_type,) + tuple(optimized_children)
    
    # Not a tuple, return as is
    return node

def strength_reduction(node):
    """Performs strength reduction optimizations like replacing multiplication
    by powers of 2 with shifts, division by power of 2, etc."""
    if isinstance(node, tuple):
        node_type = node[0]
        
        # Recursively optimize children first
        optimized_children = [strength_reduction(child) for child in node[1:]]
        
        if node_type == 'binop':
            op, left, right = optimized_children
            
            # Multiplication by power of 2 -> Left shift
            if op == '*' and isinstance(right, tuple) and right[0] == 'number':
                value = right[1]
                if value > 0 and (value & (value - 1)) == 0:  # Is power of 2
                    shift_amount = 0
                    while value > 1:
                        value >>= 1
                        shift_amount += 1
                    return ('binop', '<<', left, ('number', shift_amount))
            
            # Division by power of 2 -> Right shift
            elif op == '/' and isinstance(right, tuple) and right[0] == 'number':
                value = right[1]
                if value > 0 and (value & (value - 1)) == 0:  # Is power of 2
                    shift_amount = 0
                    while value > 1:
                        value >>= 1
                        shift_amount += 1
                    return ('binop', '>>', left, ('number', shift_amount))
            
            # x * 0 -> 0
            elif op == '*' and isinstance(right, tuple) and right[0] == 'number' and right[1] == 0:
                return ('number', 0)
            
            # x * 1 -> x
            elif op == '*' and isinstance(right, tuple) and right[0] == 'number' and right[1] == 1:
                return left
            
            # x + 0 -> x
            elif op == '+' and isinstance(right, tuple) and right[0] == 'number' and right[1] == 0:
                return left
            
            # 0 + x -> x
            elif op == '+' and isinstance(left, tuple) and left[0] == 'number' and left[1] == 0:
                return right
            
            # x - 0 -> x
            elif op == '-' and isinstance(right, tuple) and right[0] == 'number' and right[1] == 0:
                return left
            
            # x / 1 -> x
            elif op == '/' and isinstance(right, tuple) and right[0] == 'number' and right[1] == 1:
                return left
            
            # 0 / x -> 0 (except x=0)
            elif op == '/' and isinstance(left, tuple) and left[0] == 'number' and left[1] == 0:
                return ('number', 0)
        
        # Construct the optimized node with the processed children
        return (node_type,) + tuple(optimized_children)
    
    # Not a tuple, return as is
    return node

def peephole_optimization(ast):
    """Performs peephole optimizations on instruction sequences."""
    if not isinstance(ast, tuple):
        return ast
    
    node_type = ast[0]
    
    if node_type == 'statements':
        # Process the statements list to find peephole optimization opportunities
        statements = list(ast[1:])  # Convert to list for easier manipulation
        i = 0
        
        while i < len(statements) - 1:  # -1 because we need to look ahead
            current = statements[i]
            next_stmt = statements[i + 1] if i + 1 < len(statements) else None
            
            # Pattern 1: push x; pop -> (nothing)
            if (isinstance(current, tuple) and current[0] == 'push' and
                isinstance(next_stmt, tuple) and next_stmt[0] == 'pop'):
                statements.pop(i)     # Remove push
                statements.pop(i)     # Remove pop
                continue  # Don't increment i
            
            # Pattern 2: push x; push y; swap -> push y; push x
            elif (i + 2 < len(statements) and
                  isinstance(current, tuple) and current[0] == 'push' and
                  isinstance(next_stmt, tuple) and next_stmt[0] == 'push' and
                  isinstance(statements[i + 2], tuple) and statements[i + 2][0] == 'swap'):
                statements[i], statements[i + 1] = statements[i + 1], statements[i]
                statements.pop(i + 2)  # Remove the swap
                continue  # Don't increment i
            
            # Pattern 3: push x; dup -> push x; push x
            elif (isinstance(current, tuple) and current[0] == 'push' and
                  isinstance(next_stmt, tuple) and next_stmt[0] == 'dup'):
                # Replace dup with a duplicate push instruction
                statements[i + 1] = current  # Replace dup with a copy of push
                i += 1  # Move to the next instruction
                continue
            
            # Pattern 4: push x; push y; add -> push (x+y) if both are constants
            elif (i + 2 < len(statements) and
                  isinstance(current, tuple) and current[0] == 'push' and current[1] == 'int' and
                  isinstance(next_stmt, tuple) and next_stmt[0] == 'push' and next_stmt[1] == 'int' and
                  isinstance(statements[i + 2], tuple) and statements[i + 2][0] == 'add'):
                x_val = current[2]
                y_val = next_stmt[2]
                # Replace the sequence with a single push
                statements[i] = ('push', 'int', x_val + y_val)
                statements.pop(i + 1)  # Remove push y
                statements.pop(i + 1)  # Remove add
                continue  # Don't increment i
            
            # Pattern 5: push 0; add -> (nothing)
            elif (isinstance(current, tuple) and current[0] == 'push' and current[1] == 'int' and current[2] == 0 and
                  isinstance(next_stmt, tuple) and next_stmt[0] == 'add'):
                statements.pop(i)     # Remove push 0
                statements.pop(i)     # Remove add
                continue  # Don't increment i
            
            # Pattern 6: push 0; mul -> pop; push 0
            elif (isinstance(current, tuple) and current[0] == 'push' and current[1] == 'int' and current[2] == 0 and
                  isinstance(next_stmt, tuple) and next_stmt[0] == 'mul'):
                statements[i + 1] = ('pop',)  # Replace mul with pop
                continue  # Continue with the next check
            
            # Pattern 7: push 1; mul -> (nothing)
            elif (isinstance(current, tuple) and current[0] == 'push' and current[1] == 'int' and current[2] == 1 and
                  isinstance(next_stmt, tuple) and next_stmt[0] == 'mul'):
                statements.pop(i)     # Remove push 1
                statements.pop(i)     # Remove mul
                continue  # Don't increment i
            
            # Add more patterns as needed
            
            # If no optimization was applied, move to the next instruction
            i += 1
        
        # Return the optimized statements
        return ('statements',) + tuple(statements)
    
    # Recursively apply peephole optimization to all parts of the AST
    optimized_children = [peephole_optimization(child) for child in ast[1:]]
    return (node_type,) + tuple(optimized_children)

def common_subexpression_elimination(ast):
    """Eliminates redundant computations of the same expression."""
    # This is a simplistic approach that would need to be expanded
    # for full CSE implementation
    expressions_seen = {}  # Track expressions and their temporary variables
    
    def process_node(node, expr_map):
        if not isinstance(node, tuple):
            return node, expr_map
        
        node_type = node[0]
        
        # Check for binary operations that might be common subexpressions
        if node_type == 'binop':
            # Recursively process children first
            op = node[1]
            left, expr_map = process_node(node[2], expr_map)
            right, expr_map = process_node(node[3], expr_map)
            
            # Create a representation of this expression
            expr_key = (op, str(left), str(right))
            
            # If we've seen this exact expression before
            if expr_key in expr_map:
                # Use the variable that holds the result
                return expr_map[expr_key], expr_map
            
            # Otherwise, compute and remember for future use
            result = (node_type, op, left, right)
            expr_map[expr_key] = result
            return result, expr_map
        
        # Recursively process all children
        new_children = []
        for child in node[1:]:
            processed_child, expr_map = process_node(child, expr_map)
            new_children.append(processed_child)
        
        return (node_type,) + tuple(new_children), expr_map
    
    # Process the entire AST
    optimized_ast, _ = process_node(ast, expressions_seen)
    return optimized_ast

def tail_call_optimization(ast):
    """Optimizes tail recursion by transforming it to loops."""
    if not isinstance(ast, tuple):
        return ast
    
    node_type = ast[0]
    
    if node_type == 'func_def':
        # Extract function information
        func_name = ast[1]
        params = ast[2]
        body = ast[3]
        
        # Check if function body contains tail recursion
        if isinstance(body, tuple) and body[0] == 'statements':
            statements = list(body[1:])
            
            # Look for tail recursive calls
            has_tail_recursion = False
            for i, stmt in enumerate(statements):
                if (isinstance(stmt, tuple) and stmt[0] == 'return' and
                    len(stmt) > 1 and isinstance(stmt[1], tuple) and
                    stmt[1][0] == 'call' and stmt[1][1] == func_name):
                    has_tail_recursion = True
                    # Here we would transform the tail recursion to a loop
                    # For this example, we'll just mark it
                    statements[i] = ('optimize_tail_call', stmt[1])
            
            if has_tail_recursion:
                # Build new function body with tail calls optimized
                optimized_body = ('statements',) + tuple(statements)
                return ('func_def', func_name, params, optimized_body, 'tail_opt')
    
    # Recursively optimize children
    optimized_children = [tail_call_optimization(child) for child in ast[1:]]
    return (node_type,) + tuple(optimized_children)

def loop_unrolling(ast):
    """Identifies simple loop patterns and unrolls them for performance."""
    # This is a simplified stub - real implementation would need to:
    # 1. Identify loops in the control flow
    # 2. Analyze trip count if determinable
    # 3. Unroll the loop body where beneficial
    
    # For demonstration purposes, this would be an identity function
    return ast

def function_inlining(ast, inline_threshold=10):
    """Inlines small functions at their call sites to reduce call overhead."""
    # Step 1: First pass - collect information about all functions
    function_info = {}
    
    def analyze_functions(node):
        if not isinstance(node, tuple):
            return
        
        node_type = node[0]
        
        if node_type == 'func_def':
            func_name = node[1]
            body = node[3]
            
            # Count the number of statements as a simple complexity metric
            statement_count = 0
            if isinstance(body, tuple) and body[0] == 'statements':
                statement_count = len(body) - 1  # -1 for the 'statements' tag
            
            # Store function info for inlining decisions
            function_info[func_name] = {
                'size': statement_count,
                'body': body,
                'params': node[2],
                'can_inline': statement_count <= inline_threshold
            }
        
        # Recursively check all children
        for child in node[1:]:
            if isinstance(child, tuple):
                analyze_functions(child)
    
    # Collect function information
    analyze_functions(ast)
    
    # Step 2: Second pass - inline function calls where appropriate
    def inline_calls(node):
        if not isinstance(node, tuple):
            return node
        
        node_type = node[0]
        
        if node_type == 'call':
            func_name = node[1]
            args = node[2:]
            
            # Check if this function can be inlined
            if func_name in function_info and function_info[func_name]['can_inline']:
                # Here we would substitute the function body with args replaced
                # For this example, we're just marking it
                return ('inlined_call', func_name, args)
        
        # Recursively process all children
        new_children = [inline_calls(child) for child in node[1:]]
        return (node_type,) + tuple(new_children)
    
    # Inline appropriate function calls
    return inline_calls(ast)

def optimize(ast):
    """Applies multiple optimization passes to the AST."""
    print("Running enhanced optimizer...")
    current_ast = ast
    
    # --- Pass 1: Constant Folding ---
    print("Pass 1: Advanced Constant Folding...")
    current_ast = fold_constants(current_ast)
    print("Advanced constant folding complete.")

    # --- Pass 2: Constant Propagation ---
    print("Pass 2: Constant Propagation...")
    current_ast = constant_propagation(current_ast)
    print("Constant propagation complete.")
    
    # --- Pass 3: Peephole Optimization ---
    print("Pass 3: Peephole Optimization...")
    current_ast = peephole_optimization(current_ast)
    print("Peephole optimization complete.")
    
    # --- Pass 4: Strength Reduction ---
    print("Pass 4: Strength Reduction...")
    current_ast = strength_reduction(current_ast)
    print("Strength reduction complete.")
    
    # --- Pass 5: Common Subexpression Elimination ---
    print("Pass 5: Common Subexpression Elimination...")
    current_ast = common_subexpression_elimination(current_ast)
    print("Common subexpression elimination complete.")
    
    # --- Pass 6: Function Inlining ---
    print("Pass 6: Function Inlining...")
    current_ast = function_inlining(current_ast)
    print("Function inlining complete.")
    
    # --- Pass 7: Tail Call Optimization ---
    print("Pass 7: Tail Call Optimization...")
    current_ast = tail_call_optimization(current_ast)
    print("Tail call optimization complete.")
    
    # --- Pass 8: Dead Code Elimination ---
    print("Pass 8: Dead Code Elimination...")
    current_ast = eliminate_dead_code(current_ast)
    print("Dead code elimination complete.")
    
    # --- Pass 9: Loop Unrolling ---
    print("Pass 9: Loop Unrolling...")
    current_ast = loop_unrolling(current_ast)
    print("Loop unrolling complete.")
    
    print("Enhanced optimizer finished!")
    return current_ast 