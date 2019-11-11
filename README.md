# Deep Reinforcement Learning for 5G Networks

## How to use
The code to run voice is self explanatory.

For data, start by creating a folder `figures` in the same directory as your fork.  In `environment.py` change the line `self.M_ULA` to the values of your choice.  The code expects M = 4, 8, 16, 32, and 64.

For optimal, uncomment lines 428 and 437 from `main.py`.  Comment out lines 426, 439, 440, 442.  When run is complete, rename the `figures` folder to become `figures M=m optimal` after completion, where `m` takes values of M as shown above.

For the proposed solution, uncomment lines 426, 439, 440, 442 from `main.py`.  Comment out lines 428 and 437.  When run is complete, rename the `figures` folder to become `figures M=m`.

Run the script `parse.py` in every folder `figures*` you create.  This generates a few intermediary files.

Create a folder `figures` again.  Now run `plotting.py`.

For reproducibility, please use CPU and not the GPU when running the code.

## Version history
6/28/2019 Initial code release

11/6/2019 Version 2.  Normalized the power and the convergence episodes.  I choose the episode close to the median to determine convergence.
