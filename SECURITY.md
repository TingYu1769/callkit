# 安全

## 支持的版本

只维护最新的版本。

## 报告漏洞

请不要在公开的 issue 里报告安全问题。请用 GitHub 的私密漏洞报告：在仓库的 “Security” 页面点 “Report a vulnerability”。收到后会尽快回复。

callkit 只从环境变量读密钥，写记录时会把 `x-api-key`、`authorization`、`cookie` 等请求头打码。如果你发现记录里漏出了密钥，也请按上面的方式报告。

---

Please do not report security issues in public issues. Use GitHub's private vulnerability reporting ("Security" tab → "Report a vulnerability").
