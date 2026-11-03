<details>
<summary>ENG (English Version)</summary>

# Week 01 — Docker Basics and Python Development Environment Setup

An introduction to Docker concepts, basic commands, and container-based Python development on Windows.

## 1. Docker Overview

Docker: a platform for packaging applications, libraries, and configuration into **images** and running them in isolated environments called **containers**.

Image-based environment definitions for consistent setups across computers and fewer issues caused by local environment differences.

Applications in AI model training: Python and library version management, dependency isolation, and experiment environment sharing.

### Virtual Machines vs. Containers

| Category | Virtual Machine (VM) | Container |
|---|---|---|
| Architecture | Separate guest OS on a hypervisor | Shared kernel of the container host |
| Resource usage | Relatively high due to the guest OS | Relatively low |
| Startup | Guest OS boot required | Process startup without a separate OS boot |
| Isolation | Separate OS and kernel | Isolation of processes, file systems, networks, and other resources |
| Common uses | Different operating systems and OS-level isolation | Application deployment and development or experiment environments |

> Different isolation models for VMs and containers due to kernel sharing.
> Linux container execution on Windows through a Linux environment such as WSL2, provided by Docker Desktop.

### Main Benefits of Docker

- **Environment consistency**: Reduced differences between development, testing, and deployment environments
- **Portability**: Image distribution and execution across compatible Docker environments
- **Fast startup**: Rapid container creation and startup
- **Dependency isolation**: Separate libraries and runtime environments for each project

---

## 2. Core Docker Components

| Component | Description |
|---|---|
| Docker Engine | Creation, execution, and management of images, containers, and related resources |
| Image | A template containing application code, runtimes, libraries, and dependencies |
| Container | An environment created from an image, with running and stopped states |
| Registry | An image storage and distribution service, such as Docker Hub |
| Dockerfile | A file defining image build instructions |
| Volume | Docker-managed storage for data persistence beyond container deletion |

### Image–Container Relationship

Creation of multiple containers from a single image.

```text
Dockerfile
    │ docker build
    ▼
  Image
    │ docker run
    ├── Container A
    └── Container B
```

An immutable base image during container execution, with file changes stored in each container's writable layer.

No automatic incorporation of container changes into the original image.

---

## 3. Windows Environment Preparation

**PowerShell** as the shell for Windows commands in this document.

### 3.1. WSL2 Installation and Verification

Execution in an administrator PowerShell session:

```powershell
wsl --install
```

PC restart when prompted, followed by initial setup completion.

```powershell
# Default WSL version for newly installed Linux distributions
wsl --set-default-version 2

# Installed distributions and WSL versions
wsl -l -v
```

Conversion of an existing WSL1 distribution to WSL2:

```powershell
# Example for a distribution named Ubuntu
wsl --set-version Ubuntu 2
```

> WSL2 architecture: a real Linux kernel inside a lightweight virtual machine.
> Background VM management for reduced manual configuration and maintenance.

### 3.2. Docker Desktop Configuration

1. Installation and launch of Docker Desktop for Windows
2. Activation of the WSL2-based engine
3. WSL Integration activation for the relevant distribution, if Docker access from a WSL terminal is needed
4. Verification of Linux container mode

Installation verification in PowerShell:

```powershell
docker version
docker run --rm hello-world
```

- `docker version`: Docker client and server version information
- `--rm`: Automatic container removal after exit

---

## 4. Basic Docker Commands

### 4.1. Image Search, Download, and Listing

```powershell
# Image search on Docker Hub
docker search python

# Image download
docker pull python:3.12-slim

# Local image listing
docker images
```

Common image reference format: `image:tag`.

Example: `python:3.12-slim` — a slim image from the Python 3.12 series.

Default tag when omitted: generally `latest`. Project-specific version selection still necessary, regardless of the `latest` label.

### 4.2. Container Creation and Execution

```powershell
docker run -d -p 127.0.0.1:8080:80 --name myweb nginx
```

| Option | Meaning |
|---|---|
| `-d` | Background execution |
| `-p 127.0.0.1:8080:80` | Mapping of local PC port 8080 to container port 80 |
| `--name myweb` | Container name assignment |
| `nginx` | Image selection |

Browser verification at `http://localhost:8080`.

> Port publication for local PC access through binding to `127.0.0.1`.
> Default publication on all host interfaces when the host IP is omitted.

### 4.3. Container Status, Startup, and Shutdown

Individual command execution according to the required operation.

```powershell
# Running containers
docker ps

# All containers, including stopped ones
docker ps -a

# Container shutdown
docker stop myweb

# Startup of an existing stopped container
docker start myweb

# Container restart
docker restart myweb
```

- `docker run`: Creation and startup of a new container
- `docker start`: Startup of an existing container

Unique container names required; no duplicate name assignment while the existing container remains.

### 4.4. Container Shell Access and Logs

```powershell
# Shell access inside a running container
docker exec -it myweb /bin/sh

# Log output
docker logs myweb

# Real-time log monitoring
docker logs -f myweb
```

- `-i`: Standard input kept open
- `-t`: Terminal allocation
- `/bin/bash`: Alternative shell when Bash is available in the image
- `exit`: Shell termination
- `Ctrl+C`: End of real-time log monitoring

### 4.5. Container and Image Removal

```powershell
# Container shutdown and removal
docker stop myweb
docker rm myweb

# Example of unused image removal
docker rmi nginx
```

Separate operations for container removal and image removal.

### 4.6. Unused Resource Cleanup

```powershell
docker system prune
```

Cleanup targets: stopped containers, unused networks, dangling images, and unused build cache.

```powershell
docker system prune -a
```

Additional cleanup targets with `-a`: images without any associated containers.

> Potential deletion of stopped containers still needed for later use; resource review required before cleanup.
> Volume retention by default with the commands above.

---

## 5. Apache HTTP Server Practice

Official Apache HTTP Server image: `httpd`.

```powershell
docker pull httpd:2.4
docker run -d -p 127.0.0.1:8081:80 --name apache-web httpd:2.4
```

Browser verification at `http://localhost:8081`.

```powershell
# Container status and logs
docker ps
docker logs apache-web

# Container shell access
docker exec -it apache-web /bin/sh
```

Default document directory: `/usr/local/apache2/htdocs`.

Direct file editing inside the container for testing, with loss of those changes upon container deletion.

Persistent web content management through bind mounts or file inclusion in a Dockerfile.

---

## 6. Python Development Environment Setup

### 6.1. Python Image Variants

| Example | Characteristics |
|---|---|
| `python:3.12` | General-purpose Debian-based image |
| `python:3.12-slim` | Smaller image focused on Python runtime requirements |
| `python:3.12-alpine` | Alpine Linux-based image; package compatibility and build considerations due to musl libc |
| `python:<version>-windowsservercore` | Image family for Windows container environments |

Possible need for additional compilers or system libraries in `slim` images, depending on package requirements.

Image for this tutorial: `python:3.12-slim`.

Framework-specific Python compatibility verification before use in an actual AI project.

### 6.2. Container Execution with a Project Folder

Execution from the project folder in PowerShell:

```powershell
docker run -it --name python-dev -v "${PWD}:/workspace" -w /workspace python:3.12-slim bash
```

| Option | Meaning |
|---|---|
| `-it` | Interactive execution with a terminal |
| `-v "${PWD}:/workspace"` | Mounting of the current PC folder at `/workspace` inside the container |
| `-w /workspace` | Container working directory assignment |
| `bash` | Shell selection |

Verification inside the container:

```bash
python --version
python -c "print('Hello, Docker!')"
```

Bash termination through `exit`, followed by container shutdown in this example.

Restart and reconnection from PowerShell:

```powershell
docker start -ai python-dev
```

### 6.3. Bind Mounts and Volumes

| Category | Bind Mount | Docker Volume |
|---|---|---|
| Storage location | User-selected host path | Docker-managed storage |
| Common uses | Source code, dataset, and output file sharing | Persistent storage for database files and other data |
| Example | `-v "${PWD}:/workspace"` | `-v mariadb-data:/var/lib/mysql` |

Storage of files under `/workspace` in the connected PC folder in the Python example above.

**Container stop**: retention of internal changes.

**Container deletion**: removal of the container's writable layer.

Mounted locations for files requiring persistence.

### 6.4. VS Code Integration

1. Installation of the **Dev Containers** extension in VS Code
2. Verification of the running state of `python-dev`
3. Selection of `Dev Containers: Attach to Running Container...` from the Command Palette
4. Connection to `python-dev` and opening of `/workspace`
5. Installation of the Python extension in the connected container environment
6. Python file creation and execution

Startup of a stopped container from a separate PowerShell terminal:

```powershell
docker start python-dev
```

---

## 7. Environment Reproduction with a Dockerfile

No automatic addition of manually installed container packages to the original image.

Dependency tracking in `requirements.txt` and Dockerfile-based image creation for easier environment sharing.

### Project Structure

```text
project/
├── Dockerfile
├── requirements.txt
└── main.py
```

### requirements.txt

Package versions verified for the project:

```text
numpy==1.26.4
```

Example version for format demonstration; adjustment required for actual project dependencies.

### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

CMD ["python", "main.py"]
```

### main.py

```python
import numpy as np

print("Hello, Docker!")
print("NumPy version:", np.__version__)
```

### Image Build and Execution

Execution from the project folder in PowerShell:

```powershell
docker build -t ai-study:week01 .
docker run --rm ai-study:week01
```

> Improved environment reproducibility through package version pinning.
> Base image digest and transitive dependency management for stricter reproducibility.
> CPU-based Python environment in this example; additional driver, runtime, and framework configuration required for GPU training.

---

## 8. Optional Practice: MariaDB Execution

Separate MariaDB container execution for projects requiring a database.

### Environment Variable File

Creation of `mariadb.env` in the project folder, with replacement of example passwords by user-selected values.

```dotenv
MARIADB_ROOT_PASSWORD=replace_with_root_password
MARIADB_ROOT_HOST=localhost
MARIADB_DATABASE=study_db
MARIADB_USER=study_user
MARIADB_PASSWORD=replace_with_user_password
```

Addition to `.gitignore` to prevent password inclusion in Git commits:

```gitignore
mariadb.env
```

### Container Execution

```powershell
docker run -d --name mariadb --env-file ./mariadb.env -p 127.0.0.1:3306:3306 -v mariadb-data:/var/lib/mysql mariadb:11.4
```

- `mariadb-data`: Volume for database file persistence
- `study_db`: Database created during initialization
- `study_user`: User with access to the designated database

Startup log verification:

```powershell
docker logs mariadb
```

Connection settings for HeidiSQL or another database client:

| Setting | Value |
|---|---|
| Host | `127.0.0.1` |
| Port | `3306` |
| User | `study_user` |
| Password | Value assigned to `MARIADB_PASSWORD` |
| Database | `study_db` |

> Initial account, password, and database environment variables applicable during the first initialization of an empty data directory.
> No automatic password changes through environment variable updates when reusing a volume containing an existing database.

</details>

<details>
<summary>KOR (한국어 버전)</summary>

# Week 01 — Docker 기초와 Python 개발 환경 구축

Docker의 주요 개념과 기본 명령어 학습 및 Windows에서의 컨테이너 기반 Python 개발 환경 구성.

## 1. Docker 개요

Docker는 애플리케이션과 실행에 필요한 라이브러리·설정 등을 **이미지(Image)** 로 패키징하고, 이를 **컨테이너(Container)** 라는 격리된 환경에서 실행하는 플랫폼.

개발 환경을 이미지로 정의하여 다른 컴퓨터에서도 일관된 환경을 구성하고, 로컬 환경 차이로 발생하는 문제를 줄이는 데 활용.

AI 모델 학습에서는 Python·라이브러리 버전 관리 및 실험 환경 공유에 활용 가능.

### 가상머신과 컨테이너의 차이

| 구분 | 가상머신(VM) | 컨테이너 |
|---|---|---|
| 실행 구조 | 하이퍼바이저 위에서 별도의 게스트 OS 실행 | 컨테이너 실행 환경의 OS 커널 공유 |
| 리소스 사용 | 게스트 OS 포함으로 상대적으로 큰 사용량 | 상대적으로 적은 사용량 |
| 시작 과정 | 게스트 OS 부팅 필요 | 별도의 OS 부팅 없이 프로세스 실행 |
| 격리 방식 | 별도의 OS·커널을 통한 격리 | 프로세스·파일 시스템·네트워크 등의 격리 |
| 활용 예시 | 서로 다른 OS 운영, OS 단위 환경 분리 | 애플리케이션 배포, 개발·실험 환경 구성 |

> 컨테이너는 커널을 공유하므로 VM과 다른 격리 구조 사용.
> Windows에서 Docker Desktop으로 Linux 컨테이너 실행 시 WSL2 등의 Linux 실행 환경 활용.

### Docker의 주요 장점

- **환경 일관성**: 개발·테스트·배포 환경 간 차이 감소
- **이식성**: 호환되는 Docker 실행 환경에서 이미지 배포 및 실행 가능
- **빠른 실행**: 컨테이너의 신속한 생성 및 시작
- **의존성 분리**: 프로젝트별 라이브러리와 실행 환경 분리 가능

---

## 2. Docker의 핵심 구성 요소

| 구성 요소 | 설명 |
|---|---|
| Docker Engine | 이미지·컨테이너 및 관련 리소스의 생성·실행·관리 담당 |
| Image | 애플리케이션 코드, 런타임, 라이브러리 등을 포함한 실행 템플릿 |
| Container | 이미지 기반으로 생성된 실행 환경. 실행 또는 중지 상태로 존재 가능 |
| Registry | 이미지 저장·배포 서비스. 대표적인 예시로 Docker Hub 존재 |
| Dockerfile | 이미지 생성 절차를 정의한 파일 |
| Volume | 컨테이너 삭제 이후에도 데이터 보존이 가능한 Docker 관리 저장소 |

### 이미지와 컨테이너의 관계

하나의 이미지로 여러 컨테이너 생성 가능.

```text
Dockerfile
    │ docker build
    ▼
  Image
    │ docker run
    ├── Container A
    └── Container B
```

컨테이너 실행 중에도 이미지 자체는 변경되지 않는 구조. 컨테이너 내부 파일 수정 사항은 해당 컨테이너의 쓰기 가능한 계층에 저장.

컨테이너 내부 변경 사항의 원본 이미지 자동 반영은 지원하지 않음.

---

## 3. Windows 개발 환경 준비

이 문서의 Windows 명령어는 **PowerShell 기준**.

### 3.1. WSL2 설치 및 확인

관리자 권한 PowerShell에서 다음 명령어 실행.

```powershell
wsl --install
```

설치 과정에서 재시작 안내 시 PC 재시작 후 초기 설정 완료.

```powershell
# 새로 설치하는 Linux 배포판의 기본 버전 설정
wsl --set-default-version 2

# 설치된 배포판과 WSL 버전 확인
wsl -l -v
```

기존 배포판이 WSL1인 경우 해당 배포판의 WSL2 변환 진행.

```powershell
# 배포판 이름이 Ubuntu인 경우
wsl --set-version Ubuntu 2
```

> WSL2는 경량 가상머신 내부에서 실제 Linux 커널을 실행하는 구조.
> 가상머신의 자동 관리를 통해 사용자의 직접적인 VM 구성·관리 부담 감소.

### 3.2. Docker Desktop 설정

1. Docker Desktop for Windows 설치 및 실행
2. WSL2 기반 엔진 사용 설정
3. WSL 터미널에서 Docker 사용 시 해당 배포판의 WSL Integration 활성화
4. Linux 컨테이너 모드 확인

설치 후 PowerShell에서 정상 동작 확인.

```powershell
docker version
docker run --rm hello-world
```

- `docker version`: Docker 클라이언트 및 서버 버전 확인
- `--rm`: 컨테이너 종료 시 해당 컨테이너 자동 삭제

---

## 4. 기본 Docker 명령어

### 4.1. 이미지 검색·다운로드·조회

```powershell
# Docker Hub에서 이미지 검색
docker search python

# 이미지 다운로드
docker pull python:3.12-slim

# 로컬 이미지 목록 확인
docker images
```

이미지 이름의 일반적인 표기 형식은 `이미지명:태그`.

예를 들어 `python:3.12-slim`은 Python 3.12 계열의 slim 이미지.

태그 생략 시 일반적으로 `latest` 사용. 단, `latest`가 모든 프로젝트에 가장 적합한 버전임을 의미하지는 않음.

### 4.2. 컨테이너 생성 및 실행

```powershell
docker run -d -p 127.0.0.1:8080:80 --name myweb nginx
```

| 옵션 | 의미 |
|---|---|
| `-d` | 백그라운드 실행 |
| `-p 127.0.0.1:8080:80` | 로컬 PC의 8080 포트와 컨테이너의 80 포트 연결 |
| `--name myweb` | 컨테이너 이름 지정 |
| `nginx` | 실행 이미지 지정 |

브라우저에서 `http://localhost:8080` 접속 후 동작 확인.

> `127.0.0.1` 바인딩 시 로컬 PC에서 접근하는 용도로 포트 공개.
> 호스트 IP 생략 시 기본적으로 모든 호스트 인터페이스에 포트 공개.

### 4.3. 상태 확인 및 시작·중지

필요한 동작에 따라 아래 명령어 개별 실행.

```powershell
# 실행 중인 컨테이너 목록 확인
docker ps

# 중지된 컨테이너를 포함한 전체 목록 확인
docker ps -a

# 컨테이너 중지
docker stop myweb

# 중지된 기존 컨테이너 시작
docker start myweb

# 컨테이너 재시작
docker restart myweb
```

- `docker run`: 새 컨테이너 생성 및 실행
- `docker start`: 이미 생성된 컨테이너 시작

동일한 이름의 컨테이너가 존재하는 경우 같은 이름을 사용한 새 컨테이너 생성 불가.

### 4.4. 컨테이너 내부 접속 및 로그 확인

```powershell
# 실행 중인 컨테이너 내부에서 셸 실행
docker exec -it myweb /bin/sh

# 로그 확인
docker logs myweb

# 로그 실시간 확인
docker logs -f myweb
```

- `-i`: 표준 입력 유지
- `-t`: 터미널 할당
- 이미지에 Bash 포함 시 `/bin/bash` 사용 가능
- `exit` 입력으로 해당 셸 종료
- `Ctrl+C` 입력으로 로그 실시간 확인 종료

### 4.5. 컨테이너 및 이미지 삭제

```powershell
# 컨테이너 중지 후 삭제
docker stop myweb
docker rm myweb

# 사용하지 않는 이미지 삭제 예시
docker rmi nginx
```

컨테이너 삭제와 이미지 삭제는 별개의 작업.

### 4.6. 사용하지 않는 리소스 정리

```powershell
docker system prune
```

중지된 컨테이너, 사용하지 않는 네트워크, dangling 이미지, 사용하지 않는 빌드 캐시 등 정리.

```powershell
docker system prune -a
```

`-a` 추가 시 어떤 컨테이너에서도 사용하지 않는 이미지까지 정리 대상에 포함.

> 필요한 중지 컨테이너도 삭제될 수 있으므로 실행 전 대상 확인 필요.
> 위 명령어는 기본적으로 볼륨을 삭제하지 않음.

---

## 5. Apache HTTP 서버 실습

Apache HTTP Server의 공식 이미지 이름은 `httpd`.

```powershell
docker pull httpd:2.4
docker run -d -p 127.0.0.1:8081:80 --name apache-web httpd:2.4
```

브라우저에서 `http://localhost:8081` 접속 후 동작 확인.

```powershell
# 상태 및 로그 확인
docker ps
docker logs apache-web

# 컨테이너 내부 접속
docker exec -it apache-web /bin/sh
```

기본 웹 문서 경로는 `/usr/local/apache2/htdocs`.

컨테이너 내부 파일의 직접 수정을 통한 동작 확인 가능. 단, 수정 사항은 컨테이너 삭제 시 함께 소멸.

웹 파일의 지속적인 관리를 위해 바인드 마운트 연결 또는 Dockerfile을 통한 파일 포함 필요.

---

## 6. Python 개발 환경 구성

### 6.1. Python 이미지 종류

| 이미지 예시 | 특징 |
|---|---|
| `python:3.12` | Debian 기반의 일반 이미지 |
| `python:3.12-slim` | Python 실행에 필요한 구성을 중심으로 크기를 줄인 이미지 |
| `python:3.12-alpine` | Alpine Linux 기반 이미지. musl libc 사용에 따른 일부 패키지 호환성·빌드 요구 사항 확인 필요 |
| `python:<버전>-windowsservercore` | Windows 컨테이너 환경용 이미지 계열 |

`slim` 이미지는 설치 패키지에 따라 컴파일러나 시스템 라이브러리 추가 필요 가능성 존재.

본 실습에서는 `python:3.12-slim` 사용. 실제 AI 프로젝트에서는 사용할 프레임워크의 Python 지원 버전 사전 확인 필요.

### 6.2. 프로젝트 폴더를 연결한 컨테이너 실행

PowerShell에서 프로젝트 폴더로 이동 후 다음 명령어 실행.

```powershell
docker run -it --name python-dev -v "${PWD}:/workspace" -w /workspace python:3.12-slim bash
```

| 옵션 | 의미 |
|---|---|
| `-it` | 터미널을 통한 대화형 실행 |
| `-v "${PWD}:/workspace"` | 현재 PC 폴더와 컨테이너의 `/workspace` 연결 |
| `-w /workspace` | 컨테이너 작업 디렉터리 지정 |
| `bash` | 실행할 셸 지정 |

컨테이너 내부에서 다음 명령어 실행.

```bash
python --version
python -c "print('Hello, Docker!')"
```

`exit` 입력 시 Bash 종료. 이 예제에서는 Bash 종료에 따라 컨테이너도 중지.

기존 컨테이너의 재시작 및 재접속 시 PowerShell에서 다음 명령어 실행.

```powershell
docker start -ai python-dev
```

### 6.3. 바인드 마운트와 볼륨

| 구분 | 바인드 마운트 | Docker 볼륨 |
|---|---|---|
| 저장 위치 | 사용자가 지정한 호스트 경로 | Docker가 관리하는 저장소 |
| 주요 용도 | 소스 코드, 데이터셋, 결과 파일 공유 | 데이터베이스 등의 데이터 보존 |
| 예시 | `-v "${PWD}:/workspace"` | `-v mariadb-data:/var/lib/mysql` |

위 Python 예제에서 `/workspace`에 저장한 파일은 연결된 PC 폴더에도 반영.

컨테이너 **중지** 시 내부 변경 사항 유지. 컨테이너 **삭제** 시 컨테이너 자체의 쓰기 계층 소멸.

보존이 필요한 파일은 마운트된 경로에 저장 필요.

### 6.4. VS Code 연동

1. VS Code에 **Dev Containers** 확장 설치
2. `python-dev` 컨테이너 실행 상태 확인
3. 명령 팔레트에서 `Dev Containers: Attach to Running Container...` 선택
4. `python-dev` 연결 후 `/workspace` 폴더 열기
5. 연결된 컨테이너 환경에 Python 확장 설치
6. Python 파일 작성 및 실행

컨테이너가 중지된 경우 별도 PowerShell에서 먼저 실행.

```powershell
docker start python-dev
```

---

## 7. Dockerfile을 통한 개발 환경 재현

컨테이너에 수동 설치한 패키지는 원본 이미지에 반영되지 않음.

필요한 라이브러리의 `requirements.txt` 기록 및 Dockerfile 기반 이미지 생성을 통한 환경 공유 편의성 향상.

### 파일 구성

```text
project/
├── Dockerfile
├── requirements.txt
└── main.py
```

### requirements.txt

프로젝트에서 검증한 패키지 버전 기록.

```text
numpy==1.26.4
```

위 버전은 작성 형식 예시. 실제 프로젝트 의존성에 맞는 수정 필요.

### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

CMD ["python", "main.py"]
```

### main.py

```python
import numpy as np

print("Hello, Docker!")
print("NumPy version:", np.__version__)
```

### 이미지 빌드 및 실행

프로젝트 폴더의 PowerShell에서 다음 명령어 실행.

```powershell
docker build -t ai-study:week01 .
docker run --rm ai-study:week01
```

> 패키지 버전 고정을 통한 환경 재현성 향상.
> 더 엄격한 재현성이 필요한 경우 베이스 이미지 digest 및 간접 의존성까지 관리 필요.
> 본 예제는 CPU 기반 Python 환경. GPU 학습 시 별도의 드라이버·런타임·프레임워크 설정 필요.

---

## 8. 선택 실습: MariaDB 실행

데이터베이스가 필요한 실습에서 MariaDB의 별도 컨테이너 실행 가능.

### 환경 변수 파일 작성

프로젝트 폴더에 `mariadb.env` 파일 생성. 예시 비밀번호는 직접 정한 값으로 변경 필요.

```dotenv
MARIADB_ROOT_PASSWORD=replace_with_root_password
MARIADB_ROOT_HOST=localhost
MARIADB_DATABASE=study_db
MARIADB_USER=study_user
MARIADB_PASSWORD=replace_with_user_password
```

비밀번호가 포함된 파일의 Git 커밋 방지를 위해 `.gitignore`에 추가.

```gitignore
mariadb.env
```

### 컨테이너 실행

```powershell
docker run -d --name mariadb --env-file ./mariadb.env -p 127.0.0.1:3306:3306 -v mariadb-data:/var/lib/mysql mariadb:11.4
```

- `mariadb-data`: 데이터베이스 파일 보존용 볼륨
- `study_db`: 초기 생성 데이터베이스
- `study_user`: 해당 데이터베이스 접근 사용자

초기 구동 로그 확인.

```powershell
docker logs mariadb
```

HeidiSQL 등 데이터베이스 클라이언트에서 다음 정보로 접속.

| 항목 | 값 |
|---|---|
| 호스트 | `127.0.0.1` |
| 포트 | `3306` |
| 사용자 | `study_user` |
| 비밀번호 | `MARIADB_PASSWORD`에 설정한 값 |
| 데이터베이스 | `study_db` |

> 초기 계정·비밀번호·데이터베이스 환경 변수는 빈 데이터 디렉터리의 최초 초기화 시 적용.
> 기존 데이터가 있는 볼륨 재사용 시 환경 변수 변경만으로 기존 비밀번호의 자동 변경은 불가.

</details>
