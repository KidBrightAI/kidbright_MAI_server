# Source this file to prep a shell for local CLI training / conversion.
#
#   source activate.sh
#   python run_train.py --project-dir ~/my_project
#
# Sets:
#   - KBMAI_OPENCV_ROOT       path to OpenCV 3.4.13 runtime (used by main.py for spnntools/onnx2ncnn)
#   - LD_LIBRARY_PATH         extends with opencv/lib + lib_extra (libprotobuf23, libIlmImf, ...)
#   - conda activate kbmai    conda env created to match the Colab ref

export KBMAI_OPENCV_ROOT="${KBMAI_OPENCV_ROOT:-$HOME/opencv-3.4.13}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${KBMAI_OPENCV_ROOT}/lib:${KBMAI_OPENCV_ROOT}/lib_extra"
export PATH="${PATH}:${KBMAI_OPENCV_ROOT}/bin"

# Initialize conda for this shell
_conda_base="${HOME}/miniforge3"
if [ -f "${_conda_base}/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${_conda_base}/etc/profile.d/conda.sh"
    conda activate kbmai
else
    echo "[activate.sh] miniforge3 not found at ${_conda_base}" >&2
fi

echo "[activate.sh] KBMAI_OPENCV_ROOT=${KBMAI_OPENCV_ROOT}"
echo "[activate.sh] conda env       = $(basename "${CONDA_PREFIX:-none}")"
echo "[activate.sh] python          = $(command -v python)"
