import subprocess
import sys
import os

# Verifică dacă CMake este instalat, dacă nu, instalează-l
def install_cmake():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cmake"])
    except subprocess.CalledProcessError:
        print("Eroare la instalarea CMake.")

# Instalează toate dependențele din requirements.txt
def install_packages():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# Funcția principală
def main():
    install_cmake()  # Instalează CMake dacă nu este prezent
    install_packages()  # Instalează pachetele din requirements.txt

if __name__ == "__main__":
    main()
