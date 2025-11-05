# 책상 정돈 상태 평가 시스템

YOLO 객체 탐지와 시각적 복잡도 분석을 활용하여 책상의 정돈 상태를 자동으로 평가하는 AI 프로젝트입니다.

## 주요 기능

- **자동 객체 탐지**: YOLOv8 모델을 사용하여 책상 위의 다양한 물건들을 자동으로 탐지
- **정돈 점수 평가**: 여러 기준을 종합하여 책상의 정돈 상태를 0-100점으로 평가
- **상세 피드백 제공**: 점수와 함께 개선 방법에 대한 구체적인 피드백 제공
- **웹 기반 인터페이스**: Gradio를 사용한 직관적이고 사용하기 쉬운 웹 인터페이스

## 탐지 가능한 객체

- **학습 도구**: 노트북(NOTEBOOK), 펜(pen), 종이(paper), 포스트잇(post-it)
- **음료**: 물병(bottle), 컵(cup)
- **전자기기**: 노트북(laptop), 마우스(mouse), 키보드(keyboard)

## 평가 기준

### 가점 요소
- 전자기기 세트 감지 (노트북, 키보드, 마우스)
- 전자기기 정렬 상태
- 물건 간 겹침 없음
- 적절한 물건 차지 비율 (30% 이하)
- 깔끔한 시각적 복잡도

### 감점 요소
- 같은 물건이 여러 개 존재
- 지저분한 물건 (종이, 포스트잇, 물병)
- 물건 간 과도한 겹침
- 너무 많은 물건 (8개 이상)
- 과도한 물건 차지 비율 (40% 이상)
- 높은 시각적 복잡도

## 🚀 설치 및 실행

### 1. 저장소 클론

```bash
git clone https://github.com/sowon2222/visionProject.git
cd visionProject
```

### 2. 가상환경 생성 및 활성화 (선택사항)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. 필요한 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. 프로그램 실행

```bash
python desk_score_gradio.py
```

실행 후 브라우저에서 표시되는 링크로 접속하거나, 공유 링크(share=True)를 통해 다른 사람과도 함께 사용할 수 있습니다.

## 필요한 패키지

- `gradio`: 웹 인터페이스 구축
- `ultralytics`: YOLOv8 모델 사용
- `opencv-python`: 이미지 처리
- `numpy`: 수치 계산
- `pydantic`: 설정 관리

## 📁 프로젝트 구조

```
visionProject/
├── desk_score_gradio.py    # 메인 애플리케이션 파일
├── last.pt                 # 학습된 YOLO 모델 (custom)
├── yolov8n.pt             # YOLOv8 기본 모델
├── requirements.txt        # 필요한 패키지 목록
├── .gitignore             # Git 제외 파일 목록
└── README.md              # 프로젝트 설명서
```

## 사용 방법

1. 웹 인터페이스에서 "책상 사진 업로드" 버튼을 클릭
2. 평가하고 싶은 책상 사진을 업로드
3. 자동으로 객체 탐지 및 점수 평가가 수행됨
4. 탐지 결과 이미지와 상세 피드백을 확인

## 기술 스택

- **딥러닝 모델**: YOLOv8 (Ultralytics)
- **이미지 처리**: OpenCV
- **웹 프레임워크**: Gradio
- **언어**: Python 

## 라이선스
이 프로젝트는 MIT 라이선스를 따릅니다.

## 작성자
- GitHub: [sowon2222](https://github.com/sowon2222)


