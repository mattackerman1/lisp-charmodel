# Source me. Toolchain under /home/box/opt (survives box restarts; /usr does not).
export PATH=/home/box/opt/sbcl/bin:/home/box/opt/pyenv/bin:$PATH
export GPT_OPENBLAS=/home/box/opt/openblas/lib/libopenblas.so.0
export SBCL=/home/box/opt/sbcl/bin/sbcl
export PY=/home/box/opt/pyenv/bin/python
export EFF=/home/box/lisp-charmodel/efficiency
