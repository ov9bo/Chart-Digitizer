# Security policy

Chart Digitizer runs locally. The CLI makes no network connections. The web app (`digitize-ui`)
serves on `127.0.0.1` only and rejects requests whose `Host` or `Origin` header is not the local
page, so other websites open in your browser can't call it.

If you find a way around that, or any other vulnerability, please report it privately through
[GitHub's private vulnerability reporting](https://github.com/ov9bo/Chart-Digitizer/security/advisories/new)
rather than in a public issue.

Binding the web app to a non-loopback address with `--host` exposes it to your network. Only do
that on a network you trust.
