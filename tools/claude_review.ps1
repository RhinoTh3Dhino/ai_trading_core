param([string]$Prompt = "Review staged diff og foreslå meningsfulde commits")
claude -p $Prompt --permission-mode plan --allowedTools Read --max-turns 2 --output-format json
