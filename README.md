# Dalix
Parallelization for Dalix Algorithm

##Source Codes, OpenMP
- [src](src), source code folder

- [src/CMakeLists.txt](src/CMakeLists.txt), cmake build config file

- build steps

```zsh
mkdir cmake-build-debug
cd cmake-build-debug
cmake ..
make
```

- usage, enter into the folder, in which the executable file dalix is

```zsh
./dalix ../../datasets/1AJ3.pdb ../../datasets/1b71.pdb
```

- still some bugs, which lead to corruption of program

```zsh
Allocation time: 1652
^C[1]    26025 segmentation fault (core dumped)  ./dalix ../../datasets/1AJ3.pdb ../../datasets/1b71.pdb
```
