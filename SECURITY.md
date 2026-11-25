# Security Policy

## Reporting a vulnerability

Report a suspected vulnerability privately through GitHub's private vulnerability reporting: open the repository's **Security** tab and choose **Report a vulnerability**. If you cannot use it, email security@aiuc.com. Do not open a public issue, pull request or discussion about it.

Describe the affected commit or release, how to reproduce the problem and its expected impact. Leave out secrets such as provider API keys. The maintainers reply in the private report and publish a GitHub security advisory once a fix is available.

## Scope

Reports can cover the code, workflows and dependency pins in this repository, such as the handling of provider API keys, the LLM transport, the command-line interface and the release workflow. Report a vulnerability in a dependency or a third-party service to its maintainers. A concern about the model's risk estimates is not a security issue; open an ordinary issue for it.

The software is experimental, so fixes land on `main` and in the next release; earlier releases are not patched.
