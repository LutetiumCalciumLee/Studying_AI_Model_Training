# Docker and Basic Commands 
---
## VM vs. Containers
- Resource efficiency: VMs include full guest OSes; containers share the host kernel and are lighter and faster.
- Isolation: Containers use logically separated host resources and run independently.
- Startup speed: Containers start almost instantly without booting a separate OS.
- Overhead: VMs incur hypervisor/guest OS overhead; containers have much less.

## Core Components
- Docker Engine: Creates and manages containers (client–server application).
- Docker Image: Immutable bundle with app code, runtime, libraries, and tools.
- Docker Container: Running instance with its own filesystem, network stack, and process space; CPU/memory limits configurable.
- Docker Registry: Image storage and distribution (e.g., Docker Hub).

## Key Benefits
- Consistent environments across dev/test/prod, portability, fast deployment, isolation for security/resource control, and easy scaling for microservices.

## WSL2 Setup (Windows)
- WSL enables running GNU/Linux tools on Windows without full VMs or dual boot.
- Requirements: Windows 11 or Windows 10 v1903 build 18362+. Check with Win+R → “winver.”
- Steps: Run PowerShell as Administrator → “wsl --install” → reboot → set “wsl --set-default-version 2” → verify with “wsl -l -v.”

## Docker Desktop
- Download and install Docker Desktop to manage containers, images, volumes, and development environments.

## Hands-on: Apache HTTP Server
- Pull and list: “docker pull httpd”; “docker images.”
- Run and test: “docker run -d -p 80:80 --name myweb httpd:latest”; open 127.0.0.1; check with “docker ps.”
- Stop/remove: “docker stop myweb”; “docker rm myweb.”
- Bind mount site: “docker run -d -p 1234:80 -v d:\dlproject:/usr/local/apache2/htdocs --name myweb httpd:latest”; save index.html and open 127.0.0.1:1234.
- Edit in container: “docker exec -it myweb /bin/bash”; apt update; install vim; edit htdocs/index.html; exit.

## Hands-on: MariaDB Server
- Run: “docker run --name mariadb -d -p 3306:3306 -e MYSQL_ROOT_PASSWORD=<password> mariadb.”
- Configure: exec into container; apt update; install vim; set “bind-address = 0.0.0.0” in 50-server.cnf.
- Grant access: “mariadb -u root -p”; grant all privileges to 'root'@'%' with password; “flush privileges.”
- Verify with HeidiSQL client connection.

## Python Dev Container
- Run: “docker run -d -it -p 3000:3000 -v d:\dlproject\python:/workspace --name devpython python:3.12-slim.”
- Base image options: python:X.Y (larger, Debian-based), X.Y-slim (smaller, install build deps), X.Y-alpine (very small, potential compat issues), X.Y-windowsservercore (Windows-only, large).

## VS Code Integration
- Install extensions: Docker and Dev Containers.
- Attach VS Code to the devpython container, open /workspace, install the Python extension if prompted, and run a quick “print('hello world')” test.
