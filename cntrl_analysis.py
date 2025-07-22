import numpy as np

# Names of joints (must match number of values per line, which is 20 here)
joint_names = [
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_pitch",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
    "right_knee", "right_ankle_pitch",
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
    "left_knee", "left_ankle_pitch"
]

def parse_line(line):
    # Extract content inside brackets
    if "[" in line and "]" in line:
        data_str = line.split("[", 1)[1].rsplit("]", 1)[0]
        numbers = list(map(float, data_str.split()))
        return numbers if len(numbers) == 20 else None
    return None

def print_table(rad_vals, deg_vals):
    print(f"{'Joint Name':<24} | {'Radians':>10} | {'Degrees':>10}")
    print("-" * 50)
    for name, rad, deg in zip(joint_names, rad_vals, deg_vals):
        print(f"{name:<24} | {rad:10.6f} | {deg:10.3f}")
    print("\n" + "=" * 50 + "\n")

def main():
    with open("cntrl_data.txt", "r") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        qpos = parse_line(line)
        if qpos:
            qpos = np.array(qpos)
            degrees = np.degrees(qpos)
            print(f"Step {idx}:\n")
            print_table(qpos, degrees)

if __name__ == "__main__":
    main()
