# deploy.ps1
npm install
Write-Host "正在启动 Mofas-UI 应用..."
$env:DEBUG = "mofas-ui:*"
npm start