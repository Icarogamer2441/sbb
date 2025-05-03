import argparse
import sys
import os
from .parser_lexer import parse_sbb  # We'll uncomment these later
from .codegen import generate_assembly

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
