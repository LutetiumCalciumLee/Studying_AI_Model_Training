<details>
<summary>ENG (English Version)</summary>

## **Docker & AI Model Training**

### **VM vs. Containers**
- **Efficiency**: Containers are lighter and faster than VMs since they share the host kernel.
- **Isolation**: Both offer isolation, but containers are more resource-efficient.
- **Startup Speed**: Containers launch almost instantly.
- **Overhead**: VMs have more overhead due to full OS/hypervisor.

### **Core Docker Components**
- **Docker Engine**: Manages containers.
- **Image**: Blueprint containing code, dependencies.
- **Container**: Running instance.
- **Registry**: Stores and distributes images (e.g., Docker Hub).

### **Benefits of Docker**
- Reproducible environments across stages (dev → prod)
- Portability & fast deployment
- Resource isolation & security
- Scalable for microservices

### **WSL2 for Windows**
- Allows Linux tools on Windows without full VM.
- Install via PowerShell: `wsl --install`, set version 2, check with `wsl -l -v`.

### **Docker Desktop**
- GUI for managing containers, volumes, networks, and images.

### **Apache HTTP Server (Hands-on)**
- Pull: `docker pull httpd`
- Run: `docker run -d -p 80:80 --name myweb httpd:latest`
- Edit via container: Use `docker exec` and install editors (e.g., vim).

### **MariaDB Setup**
- Run: `docker run --name mariadb ...`
- Configure: Modify `bind-address`, grant root access.
- Connect with external clients like HeidiSQL.

### **Python Development Environment**
- Run dev container: `docker run -d -it ... python:3.12-slim`
- Image options:
  - `python:X.Y`: Full Debian base
  - `X.Y-slim`: Minimal, requires build tools
  - `X.Y-alpine`: Lightweight, less compatible
  - `X.Y-windowsservercore`: Windows only

### **VS Code Integration**
- Extensions: Docker + Dev Containers
- Attach VS Code to container, install Python extension, test with print script.

</details>

<details>
<summary>KOR (한국어 버전)</summary>

## **도커 및 AI 모델 학습 환경**

### **가상머신(VM) vs. 컨테이너**
- **효율성**: 컨테이너는 호스트 커널을 공유하므로 더 가볍고 빠름.
- **격리성**: VM과 마찬가지로 독립성 보장하지만 리소스 효율 우수.
- **부팅 속도**: 컨테이너는 즉시 실행됨.
- **오버헤드**: VM은 하이퍼바이저와 OS로 인해 무거움.

### **도커의 핵심 구성 요소**
- **도커 엔진**: 컨테이너 실행·관리
- **이미지**: 앱 코드 및 라이브러리 포함된 불변 템플릿
- **컨테이너**: 이미지의 실행 인스턴스
- **레지스트리**: 이미지 저장소 (예: Docker Hub)

### **도커의 주요 장점**
- 개발~배포까지 동일 환경 제공
- 이식성 우수 및 빠른 배포
- 리소스/보안 격리
- 마이크로서비스에 적합한 확장성

### **WSL2 설정 (Windows)**
- Windows에서 전체 VM 없이 리눅스 사용 가능.
- PowerShell에서 설치: `wsl --install`, 버전 2로 설정 후 `wsl -l -v`로 확인.

### **Docker Desktop**
- GUI 기반으로 이미지, 컨테이너, 볼륨, 네트워크 등을 관리.

### **Apache HTTP 서버 실습**
- 가져오기: `docker pull httpd`
- 실행: `docker run -d -p 80:80 --name myweb httpd:latest`
- 편집: `docker exec` 후 vim 등 설치하여 파일 수정 가능.

### **MariaDB 서버 설정**
- 실행: `docker run --name mariadb ...`
- 설정: `bind-address` 수정, 루트 계정 외부 접근 허용.
- HeidiSQL 등 외부 툴로 접속 테스트 가능.

### **파이썬 개발용 컨테이너**
- 실행: `docker run -d -it ... python:3.12-slim`
- 이미지 종류:
  - `python:X.Y`: 기본 Debian 이미지
  - `X.Y-slim`: 최소 이미지 (빌드 툴 필요)
  - `X.Y-alpine`: 매우 가볍지만 호환성 주의
  - `X.Y-windowsservercore`: 윈도우 전용, 크기 큼

### **VS Code 연동**
- 확장 설치: Docker, Dev Containers
- 컨테이너에 연결 → 작업 폴더 열기 → Python 확장 설치 후 테스트 코드 실행

</details>
