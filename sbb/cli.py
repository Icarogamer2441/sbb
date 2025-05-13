import argparse
import sys
import os
from .parser_lexer import parse_sbb, expand_macros_in_ast, ImportDecl, Program, MacroDef, FuncDecl, DataDecl # Added ImportDecl and other AST nodes
from .codegen import generate_assembly

# --- Import Resolution ---
def resolve_imports(current_ast, current_file_path, processed_files):
    """
    Resolves import statements in the AST.
    - current_ast: The AST of the current file being processed.
    - current_file_path: Absolute path to the current file.
    - processed_files: A set of absolute file paths already processed to prevent circular imports.
    Returns a new list of declarations with imports resolved and replaced.
    """
    if not isinstance(current_ast, Program):
        return current_ast # Should not happen if called correctly

    if current_file_path in processed_files:
        # print(f"Note: Skipping already processed file (circular import detected or redundant import): {current_file_path}", file=sys.stderr)
        # Return an empty list of declarations for a file that's already been processed in the current chain
        # This effectively means its declarations are already accounted for higher up the import chain.
        return []


    processed_files.add(current_file_path)
    # print(f"Processing imports for: {current_file_path}")

    final_declarations = []
    current_dir = os.path.dirname(current_file_path)

    for decl in current_ast.declarations:
        if isinstance(decl, ImportDecl):
            imported_file_relative_path = decl.filepath
            # Ensure the imported path is treated as relative to the current file's directory
            imported_file_abs_path = os.path.abspath(os.path.join(current_dir, imported_file_relative_path))

            if not os.path.exists(imported_file_abs_path):
                raise FileNotFoundError(f"Error: Imported file '{imported_file_abs_path}' (from '{imported_file_relative_path}' in '{current_file_path}') not found.")

            # print(f"  Importing: {imported_file_abs_path}")
            try:
                with open(imported_file_abs_path, 'r') as f:
                    imported_source_code = f.read()
            except IOError as e:
                raise IOError(f"Error reading imported file '{imported_file_abs_path}': {e}")

            imported_ast = parse_sbb(imported_source_code)
            if not imported_ast:
                raise SyntaxError(f"Parsing failed for imported file: {imported_file_abs_path}")

            # Recursively resolve imports for the newly parsed AST
            # Pass a copy of processed_files to handle different import branches correctly,
            # or manage it carefully if passed by reference. For simplicity, a copy is safer here
            # to avoid a file being marked as processed globally when it's only processed in one branch.
            # However, for strict circular dependency detection, the original set should be used.
            # Let's use the original set to correctly detect circular dependencies.
            imported_declarations = resolve_imports(imported_ast, imported_file_abs_path, processed_files)
            final_declarations.extend(imported_declarations)
        else:
            # Keep non-import declarations (FuncDecl, DataDecl, MacroDef)
            final_declarations.append(decl)
    
    return final_declarations


def main():
    parser = argparse.ArgumentParser(description='Compile sbb language file to GAS x86_64 assembly or executable.',
                                     epilog="Optimizations are enabled by default. Use --no-opt to disable them.")
    parser.add_argument('input_file', help='Path to the .sbb input file')
    parser.add_argument('-o', '--output', help='Output file path. If ends with .s, outputs assembly. Otherwise, attempts to create an executable.')
    # Add the --no-opt flag
    parser.add_argument("--no-opt", action="store_true", help="Disable optimizations (which are enabled by default)")
    # Add optimization flags later, e.g., -O1, -O2, etc.
    # parser.add_argument('-O', '--optimize', type=int, default=0, choices=[0, 1, 2, 3], help='Optimization level')

    args = parser.parse_args()

    input_file = args.input_file
    output_path = args.output

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.", file=sys.stderr)
        sys.exit(1)

    if not input_file.endswith('.sbb'):
        print(f"Warning: Input file '{input_file}' does not have a .sbb extension.", file=sys.stderr)

    try:
        with open(input_file, 'r') as f:
            source_code = f.read()
    except IOError as e:
        print(f"Error reading input file '{input_file}': {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Compiling {input_file}...")

    # 1. Parse the source code (to be implemented)
    print("Parsing...")
    ast = parse_sbb(source_code)
    if not ast:
        print("Parsing failed.", file=sys.stderr)
        sys.exit(1)
    # print("Parsing... (Not implemented yet)")
    # ast = None # Placeholder

    # Resolve imports
    print("Resolving imports...")
    try:
        # Get absolute path of the main input file for correct relative import resolution
        abs_input_file_path = os.path.abspath(input_file)
        processed_files_set = set()
        # Resolve imports. This returns a new list of all declarations.
        all_declarations = resolve_imports(ast, abs_input_file_path, processed_files_set)
        # Update the AST's declarations. The original AST object (Program) is preserved.
        ast.declarations = all_declarations
        print("Import resolution complete.")
    except (FileNotFoundError, SyntaxError, IOError, ValueError) as e: # Catch specific errors
        print(f"Error during import resolution: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e: # Catch any other unexpected errors
        print(f"An unexpected error occurred during import resolution: {e}", file=sys.stderr)
        # import traceback
        # traceback.print_exc(file=sys.stderr) # For more detailed debugging if needed
        sys.exit(1)


    # Expand macros (after imports are resolved and all declarations are collected)
    print("Expanding macros...")
    try:
        defined_macros_map = {} # This map will be populated by expand_macros_in_ast
        # expand_macros_in_ast modifies the ast in-place
        ast = expand_macros_in_ast(ast, defined_macros_map)
        print("Macro expansion complete.")
    except ValueError as e: # Catch specific errors from expansion
        print(f"Error during macro expansion: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e: # Catch any other unexpected errors
        print(f"An unexpected error occurred during macro expansion: {e}", file=sys.stderr)
        # import traceback
        # traceback.print_exc(file=sys.stderr) # For more detailed debugging if needed
        sys.exit(1)

    # 2. Apply Optimizations (Conditionally)
    if not args.no_opt:
        print("Running optimizer...")
        # Import the optimizer function (place import at top ideally, but here for locality)
        try:
            from .optimizer import optimize
            ast = optimize(ast)
            print("Optimization complete.")
        except ImportError:
            print("Warning: Optimizer module not found or could not be imported. Skipping optimizations.", file=sys.stderr)
        except Exception as e:
            print(f"Error during optimization: {e}", file=sys.stderr)
            # Decide if you want to exit or continue without optimization
            # sys.exit(1) # Option: Exit on optimization error
            print("Continuing compilation without optimizations.")
    else:
        print("Optimizations disabled by user.")

    # 3. Generate Assembly Code (to be implemented)
    print("Generating Assembly...")
    assembly_code = generate_assembly(ast)
    # print("Generating Assembly... (Not implemented yet)")
    # assembly_code = "# Assembly generation not implemented yet\n" # Placeholder

    output_is_assembly = False
    output_is_executable = False
    base_output_name = os.path.splitext(os.path.basename(input_file))[0]
    asm_file_path = f"{base_output_name}.s"
    obj_file_path = f"{base_output_name}.o"
    exe_file_path = base_output_name

    if output_path:
        if output_path.endswith('.s'):
            output_is_assembly = True
            asm_file_path = output_path
            exe_file_path = os.path.splitext(output_path)[0]
            obj_file_path = f"{exe_file_path}.o"
        else:
            output_is_executable = True
            exe_file_path = output_path
            asm_file_path = f"{exe_file_path}.s"
            obj_file_path = f"{exe_file_path}.o"
    else:
        # Default: Create executable with the same base name
        output_is_executable = True

    # 4. Write Assembly File
    try:
        with open(asm_file_path, 'w') as f:
            f.write(assembly_code)
        print(f"Assembly code written to {asm_file_path}")
    except IOError as e:
        print(f"Error writing assembly file '{asm_file_path}': {e}", file=sys.stderr)
        sys.exit(1)

    # 5. Assemble and Link (if executable requested)
    if output_is_executable:
        print(f"Generating executable {exe_file_path}...")
        # Use gcc (or as/ld directly) to assemble and link
        # Ensure gcc is installed and in PATH
        assemble_cmd = f"gcc -no-pie -o {obj_file_path} -c {asm_file_path}" # Use -no-pie for simpler executables
        link_cmd = f"gcc -no-pie -o {exe_file_path} {obj_file_path}"

        print(f"Running: {assemble_cmd}")
        assemble_ret = os.system(assemble_cmd)
        if assemble_ret != 0:
            print(f"Assembly command failed with exit code {assemble_ret}.", file=sys.stderr)
            # Optionally keep the .s file for debugging
            sys.exit(1)

        print(f"Running: {link_cmd}")
        link_ret = os.system(link_cmd)
        if link_ret != 0:
            print(f"Link command failed with exit code {link_ret}.", file=sys.stderr)
            # Optionally keep .s and .o files
            sys.exit(1)

        print(f"Executable created at {exe_file_path}")

        # Clean up intermediate files unless only assembly was requested
        if not output_is_assembly:
            try:
                # Only remove .s if we didn't explicitly ask for it via -o *.s
                if not (output_path and output_path.endswith('.s')):
                    os.remove(asm_file_path)
                os.remove(obj_file_path)
            except OSError as e:
                print(f"Warning: Could not clean up intermediate files: {e}", file=sys.stderr)

    print("Compilation finished.")

if __name__ == '__main__':
    main()
