## scripts/clean_pycache.py

## brug: kør    python scripts/clean_pycache.py




import os
import shutil

def clean_pycache(root="."):
    n_pycache = 0
    n_pyc = 0
    for dirpath, dirnames, filenames in os.walk(root):
        # Slet alle __pycache__ mapper
        if "__pycache__" in dirnames:
            pycache_path = os.path.join(dirpath, "__pycache__")
            try:
                shutil.rmtree(pycache_path)
                print(f"[SLETTET] {pycache_path}")
                n_pycache += 1
            except Exception as e:
                print(f"[FEJL] Kunne ikke slette {pycache_path}: {e}")
        # Slet alle .pyc filer
        for fname in filenames:
            if fname.endswith(".pyc"):
                pyc_path = os.path.join(dirpath, fname)
                try:
                    os.remove(pyc_path)
                    print(f"[SLETTET] {pyc_path}")
                    n_pyc += 1
                except Exception as e:
                    print(f"[FEJL] Kunne ikke slette {pyc_path}: {e}")
    print(f"\nFærdig! Slettede {n_pycache} __pycache__ mapper og {n_pyc} .pyc filer.")

if __name__ == "__main__":
    clean_pycache(".")
