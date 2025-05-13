sbb examples/inc_dec.sbb
sbb examples/variables.sbb
sbb examples/hello.sbb
sbb examples/funcs.sbb
sbb examples/if-else-labels.sbb
sbb examples/memory.sbb
sbb examples/stack_ops.sbb
sbb examples/macros.sbb
sbb examples/import_test/imp_main.sbb

echo "Running inc_dec"
./inc_dec
echo ""
echo "Running variables"
./variables
echo ""
echo "Running hello"
./hello
echo ""
echo "Running funcs"
./funcs
echo ""
echo "Running if-else-labels"
./if-else-labels
echo ""
echo "Running memory"
./memory
echo ""
echo "Running stack_ops"
./stack_ops
echo ""
echo "Running macros"
./macros
echo ""
echo "Running imp_main"
./imp_main
echo ""

rm ./inc_dec
rm ./variables
rm ./hello
rm ./funcs
rm ./if-else-labels
rm ./memory
rm ./stack_ops
rm ./macros
rm ./imp_main