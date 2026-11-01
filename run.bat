@echo off
rem ============================================================
rem  Запуск «СтройКонтур» из корня папки "Хакатон".
rem  Сам приложение лежит в PythonProject\run.bat, здесь только
rem  ярлык, чтобы не искать его по папкам.
rem ============================================================
setlocal
cd /d "%~dp0PythonProject"

if not exist "run.bat" (
    echo [ERROR] Не найден файл PythonProject\run.bat
    echo Проверьте, что папка PythonProject лежит рядом с этим файлом.
    pause
    exit /b 1
)

call run.bat
