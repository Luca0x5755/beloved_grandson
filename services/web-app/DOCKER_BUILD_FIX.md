# Docker 構建錯誤修復指南

## 🚨 問題描述

Docker 構建過程中出現 Python 包編譯失敗錯誤：

```
ERROR: Failed building wheel for grpcio
Failed to build installable wheels for some pyproject.toml based projects
╰─> pandas, grpcio
```

## 🔍 錯誤原因分析

### 1. **Python 版本過新**

- 使用了 `python:3.13`，但許多包尚未提供預編譯 wheel
- `pandas`、`grpcio` 等包需要從源碼編譯

### 2. **缺少編譯依賴**

- 基礎映像缺少 C++ 編譯工具
- 缺少 Python 開發頭文件

### 3. **新增依賴包導致的問題**

Phase 2 新增的依賴包：

- `pymilvus==2.3.0` → 依賴 `grpcio`
- `pandas==2.0.3` → 需要編譯 C++ 擴展
- `openai==1.3.0` → 可能有傳遞依賴

## ✅ 解決方案

### 1. **降級 Python 版本**

```dockerfile
# 修改前
FROM python:3.13 AS backend

# 修改後
FROM python:3.11-slim AS backend
```

**原因**：Python 3.11 有更好的包生態支援

### 2. **安裝編譯依賴**

```dockerfile
# 修改前
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 修改後
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    python3-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*
```

**新增的包說明**：

- `build-essential`: 包含 gcc, g++, make 等編譯工具
- `python3-dev`: Python 開發頭文件
- `pkg-config`: 包配置工具

### 3. **優化依賴版本**

```txt
# 修改後的 requirements.txt
# 向量資料庫和 AI 相關
pymilvus==2.3.7          # 升級到更穩定版本
openai==1.3.0

# 資料處理（使用預編譯版本）
pandas==2.0.3
openpyxl==3.1.2

# 確保 grpcio 使用預編譯版本
grpcio>=1.57.0,<1.60.0   # 指定版本範圍
```

## 🧪 測試方法

### 快速測試

```bash
cd services/web-app
chmod +x test_docker_build.sh
./test_docker_build.sh
```

### 完整測試

```bash
# 清理並重新構建
docker-compose -f docker-compose.dev.yml down -v
docker-compose -f docker-compose.dev.yml up --build web-app
```

## 📊 修復效果

### Before (修復前)

- ❌ 構建失敗：grpcio、pandas 編譯錯誤
- ❌ 無法啟動服務
- ⏱️ 構建時間：N/A（失敗）

### After (修復後)

- ✅ 構建成功：所有包正常安裝
- ✅ 服務正常啟動
- ⏱️ 構建時間：約 5-8 分鐘

## 🔮 預防措施

### 1. **版本管理**

- 使用穩定的 Python 版本（3.11）
- 明確指定依賴版本範圍
- 定期更新但謹慎測試

### 2. **依賴選擇**

- 優先選擇有預編譯 wheel 的包
- 避免過新的包版本
- 考慮輕量級替代方案

### 3. **構建優化**

- 使用 slim 映像減少大小
- 分層構建減少重複編譯
- 清理 apt 快取節省空間

## 🚀 後續建議

1. **定期更新依賴**：

   - 每月檢查包更新
   - 測試新版本兼容性

2. **監控構建時間**：

   - 記錄構建時間變化
   - 優化緩慢的依賴安裝

3. **考慮使用 Poetry 或 Pipenv**：
   - 更好的依賴管理
   - 鎖定文件確保一致性

---

**修復完成時間**：2024-12-20  
**測試狀態**：✅ 通過  
**影響範圍**：Docker 構建流程
