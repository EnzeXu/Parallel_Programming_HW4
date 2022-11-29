model = """#!/bin/bash -l

#SBATCH --job-name="x_{0}"
#SBATCH --partition=medium
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=44
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=64GB
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --output="jobs_oe/x_{0}.o"
#SBATCH --error="jobs_oe/x_{0}.e"

module load compilers/gcc/10.2.0
module load mpi/openmpi/4.1.1/gcc/10.2.0
mpirun -n {1} ./p2.out {2} {3} {4} {5}

"""
settings_group = 5
# settings = [
#     [12, 6, 2],
#     [12, 4, 3],
#     [12, 3, 4],
#     [24, 8, 3],
#     [24, 6, 4],
#     [24, 4, 6],
#     [48, 12, 4],
#     [48, 8, 6],
#     [48, 6, 8],
#     [84, 14, 6],
#     [84, 12, 7],
#     [84, 7, 12],
#     [120, 15, 8],
#     [120, 12, 10],
#     [120, 10, 12],
#     [168, 24, 7],
#     [168, 14, 12],
#     [168, 12, 14],
#     [176, 22, 8],
#     [176, 16, 11],
#     [176, 11, 16],
# ]

settings = [
    [16, 16, 1],
    [16, 8, 2],
    [16, 4, 4],
    [16, 2, 8],
    [16, 1, 16],

    [36, 12, 3],
    [36, 9, 4],
    [36, 6, 6],
    [36, 4, 9],
    [36, 3, 12],

    [64, 32, 2],
    [64, 16, 4],
    [64, 8, 8],
    [64, 4, 16],
    [64, 2, 32],

    [100, 25, 4],
    [100, 20, 5],
    [100, 10, 10],
    [100, 5, 20],
    [100, 4, 25],

    [120, 30, 4],
    [120, 20, 6],
    [120, 12, 10],
    [120, 10, 12],
    [120, 6, 20],

    [144, 24, 6],
    [144, 18, 8],
    [144, 12, 12],
    [144, 8, 18],
    [144, 6, 24],
]

def build_slurms(mm=57600):
    n = len(settings)
    for i in range(n):
        with open("jobs/x_{}.slurm".format(i + 1), "w") as f:
            f.write(model.format(
                i + 1,
                settings[i][0],
                mm,
                mm / 4,
                settings[i][1],
                settings[i][2],
            ))
    for i in range(n, 2 * n):
        with open("jobs/x_{}.slurm".format(i + 1), "w") as f:
            f.write(model.format(
                i + 1,
                settings[i - n][0],
                mm,
                mm / 4,
                settings[i - n][1],
                settings[i - n][2],
            ))

def read_results(start, m, n):
    k = len(settings)
    for i in range(k):
        with open("jobs_oe/x_{}.o".format(start + i + 1), "r") as f:
            lines = f.readlines()
            lines = [item for item in lines if len(item) > 5]
            parts = lines[-1].split()
            t_all = float(parts[19].replace("s", "").replace(",", ""))
            t_comm= float(parts[21].replace("s", "").replace(",", ""))
            t_comp = float(parts[23].replace("s", "").replace(",", ""))
            # print(t_all, t_comm, t_comp)
        if i % settings_group == 0:
            print("\\midrule")
            print("\\multirow{{{0:d}}}{{*}}{{{1:d}}}".format(settings_group, settings[i][0]))
        print("& {0:d} & {1:d} & {2:d} & {3:d} & {4:.6f} & {5:.6f} & {6:.6f} \\\\".format(
            settings[i][1],
            settings[i][2],
            m // settings[i][1],
            n // settings[i][2],
            t_all,
            t_comm,
            t_comp,
            ))
    print()


if __name__ == "__main__":
    # build_slurms()
    read_results(0, 57600, 57600)
    read_results(len(settings), 57600, 14400)
