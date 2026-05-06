Set WshShell = WScript.CreateObject("WScript.Shell")
strDesktop = WshShell.SpecialFolders("Desktop")
Set Shortcut = WshShell.CreateShortcut(strDesktop & "\Clawdbot Trader.lnk")
Shortcut.TargetPath = "C:\clawdtrader\START_TRADER.bat"
Shortcut.WorkingDirectory = "C:\clawdtrader"
Shortcut.IconLocation = "C:\clawdtrader\assets\clawdbot.ico"
Shortcut.Description = "Clawdbot Live Trader (DeepSeek + Alpaca)"
Shortcut.Save()
WScript.Echo "Desktop shortcut created on " & strDesktop
