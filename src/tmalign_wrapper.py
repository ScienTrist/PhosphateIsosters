import subprocess
import os
import re
import platform
import time

from utils import to_wsl_path

def run_tmalign(pdb1_path, pdb2_path, tmalign_executable="./TMalign", fast_mode=True):
    """
    Runs TMalign to compare two PDB files and returns the TM-score and RMSD.
    - fast_mode: If True, uses the '-fast' flag (10x faster, <1% accuracy loss).
    """
    is_windows = (platform.system() == "Windows")
    
    # Use a simple filename for the matrix
    matrix_filename = f"matrix_{os.getpid()}_{int(time.time())}.txt"
    matrix_path_win = os.path.abspath(matrix_filename)
    
    fast_flag = "-fast" if fast_mode else ""
    
    if is_windows:
        p1 = to_wsl_path(os.path.abspath(pdb1_path))
        p2 = to_wsl_path(os.path.abspath(pdb2_path))
        m_path_wsl = to_wsl_path(os.path.abspath(matrix_path_win))
        t_exec = to_wsl_path(os.path.abspath(tmalign_executable))

        
        # COMBINED COMMAND: Run alignment, cat the matrix, and rm the matrix in ONE WSL call
        combined_linux_cmd = f"'{t_exec}' '{p1}' '{p2}' {fast_flag} -m '{m_path_wsl}'; cat '{m_path_wsl}'; rm '{m_path_wsl}'"
        command = ["wsl", "sh", "-c", combined_linux_cmd]
    else:
        # NATIVE LINUX: Same combined logic but no 'wsl' prefix
        p1 = os.path.abspath(pdb1_path)
        p2 = os.path.abspath(pdb2_path)
        m_path = os.path.abspath(matrix_path_win)
        combined_linux_cmd = f"'{tmalign_executable}' '{p1}' '{p2}' {fast_flag} -m '{m_path}'; cat '{m_path}'; rm '{m_path}'"
        command = ["sh", "-c", combined_linux_cmd]

    try:
        # On both platforms, the stdout now contains the full TM-align output followed by the matrix
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout
        matrix_content = output 

        # Parse TM-score and RMSD
        rmsd_match = re.search(r"RMSD=\s+([\d.]+)", output)
        tm_score_1_match = re.search(r"TM-score=\s+([\d.]+)\s+\(if normalized by length of Chain_1", output)
        tm_score_2_match = re.search(r"TM-score=\s+([\d.]+)\s+\(if normalized by length of Chain_2", output)
        
        transformation = None

        if matrix_content:
            # Updated regex: more flexible with leading whitespace
            m_lines = re.findall(r"^\s*\d\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)", matrix_content, re.MULTILINE)
            if len(m_lines) == 3:
                t = [float(line[0]) for line in m_lines]
                u = [[float(line[1]), float(line[2]), float(line[3])] for line in m_lines]
                transformation = {"t": t, "u": u}
        
        return {
            "rmsd": float(rmsd_match.group(1)) if rmsd_match else None,
            "tm_score_1": float(tm_score_1_match.group(1)) if tm_score_1_match else None,
            "tm_score_2": float(tm_score_2_match.group(1)) if tm_score_2_match else None,
            "transformation": transformation,
            "raw_output": output
        }
        
    except subprocess.CalledProcessError as e:
        return None
    except Exception:
        return None

if __name__ == "__main__":
    pass
