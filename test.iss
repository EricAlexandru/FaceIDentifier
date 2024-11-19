[Setup]
AppName=FaceIdentifier
AppVersion=1.0
DefaultDirName={pf}\FaceIdentifier
DefaultGroupName=FaceIdentifier
OutputDir=.\Output
OutputBaseFilename=FaceIdentifier_Installer
Compression=lzma
SolidCompression=yes

[Files]
; Include fișierul EXE creat de PyInstaller
Source: "dist\FaceIDentifier.exe"; DestDir: "{app}"; Flags: ignoreversion
; Include fișierul requirements.txt și install.py
Source: "requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "install.py"; DestDir: "{app}"; Flags: ignoreversion

[Run]
; Rulează scriptul Python de instalare
Filename: "{app}\python.exe"; Parameters: "{app}\install.py"; Flags: runhidden
