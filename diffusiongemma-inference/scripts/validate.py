import bench
print(">>> MATH500 think=OFF"); bench.run("math500", n=10, bs=4, max_new_tokens=1024, steps=48, think_override=False, tag="_valoff")
print(">>> MATH500 think=ON");  bench.run("math500", n=10, bs=4, max_new_tokens=1024, steps=48, think_override=True,  tag="_valon")
print(">>> MBPP think=OFF");    bench.run("mbpp",    n=10, bs=4, max_new_tokens=768,  steps=48, think_override=False, tag="_val")
print(">>> SUDOKU think=ON");   bench.run("sudoku",  n=6,  bs=4, max_new_tokens=768,  steps=48, think_override=True,  tag="_valon")
print(">>> SUDOKU think=OFF");  bench.run("sudoku",  n=6,  bs=4, max_new_tokens=512,  steps=48, think_override=False, tag="_valoff")
print("VALIDATE_DONE")
