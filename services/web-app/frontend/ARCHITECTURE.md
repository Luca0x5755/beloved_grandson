# Frontend 架構說明

## 📁 目錄結構

```
frontend/
├── src/
│   ├── apps/                    # 應用程式分離
│   │   ├── liff/                # LIFF 病患端應用
│   │   │   ├── pages/           # LIFF 頁面
│   │   │   ├── components/      # LIFF 專用元件
│   │   │   └── routes.jsx       # LIFF 路由配置
│   │   │
│   │   └── dashboard/           # Dashboard 治療師端應用
│   │       ├── pages/           # Dashboard 頁面
│   │       ├── components/      # Dashboard 專用元件
│   │       ├── layouts/         # Dashboard 佈局元件
│   │       └── routes.jsx       # Dashboard 路由配置
│   │
│   ├── shared/                  # 共用資源
│   │   ├── api/                 # API 客戶端和 hooks
│   │   ├── contexts/            # React Context (Auth, Theme)
│   │   ├── components/          # 共用元件
│   │   ├── utils/               # 工具函式和 mock 資料
│   │   └── config.js            # 全域配置
│   │
│   ├── assets/                  # 靜態資源
│   ├── styles/                  # 全域樣式
│   └── App.jsx                  # 主應用程式進入點
```

## 🎯 架構優勢

### 1. 清晰的關注點分離

- **LIFF 應用**：專注於病患端功能（問卷、語音對話、健康記錄）
- **Dashboard 應用**：專注於治療師端管理（個案管理、數據分析、任務管理）
- **Shared 資源**：避免重複代碼，統一管理共用邏輯

### 2. 獨立開發和部署

- 兩個應用可以獨立開發，不會互相干擾
- 可以有不同的設計風格和 UI 框架
- 未來可以輕鬆分離成兩個獨立專案

### 3. 更好的程式碼組織

- 每個應用有自己的路由系統
- 專屬元件和頁面清楚分類
- 共用邏輯集中管理

## 🚀 路由架構

### LIFF 路由 (`/liff/*`)

```
/liff                    # LIFF 主頁（Dashboard）
/liff/voice-chat         # 語音對話
/liff/daily-metrics      # 每日健康記錄
/liff/questionnaire/cat  # CAT 問卷
/liff/questionnaire/mmrc # mMRC 問卷
/liff/register           # 註冊頁面
```

### Dashboard 路由 (`/dashboard/*`)

```
/dashboard/overview      # 總覽頁
/dashboard/cases         # 個案列表
/dashboard/cases/:id     # 個案詳情
/dashboard/education     # 衛教資源
/dashboard/tasks         # 任務管理
```

## 🔧 開發指南

### 新增 LIFF 頁面

1. 在 `src/apps/liff/pages/` 建立新頁面元件
2. 在 `src/apps/liff/routes.jsx` 註冊路由
3. 專屬元件放在 `src/apps/liff/components/`

### 新增 Dashboard 頁面

1. 在 `src/apps/dashboard/pages/` 建立新頁面元件
2. 在 `src/apps/dashboard/routes.jsx` 註冊路由
3. 專屬元件放在 `src/apps/dashboard/components/`

### 共用資源

- API hooks: `src/shared/api/hooks.js`
- 工具函式: `src/shared/utils/`
- 共用元件: `src/shared/components/`
- 全域配置: `src/shared/config.js`

## 📝 注意事項

1. **Import 路徑**：使用相對路徑時要注意目錄層級
2. **樣式隔離**：考慮使用 CSS Modules 或 styled-components 避免樣式衝突
3. **狀態管理**：使用 Context API 進行跨元件狀態共享
4. **權限控制**：Dashboard 需要 `is_staff` 權限，LIFF 開放給一般用戶

## 🎨 設計原則

### LIFF 應用

- 行動優先設計
- 簡潔直觀的介面
- 大按鈕和清晰的文字
- 支援 LINE 內嵌瀏覽器

### Dashboard 應用

- 桌面優先，響應式設計
- 資料視覺化為主
- 專業的管理介面
- 豐富的互動功能

## 🚦 部署策略

### 開發環境

```bash
npm run dev
# 訪問 http://localhost:3000/liff (病患端)
# 訪問 http://localhost:3000/dashboard (治療師端)
```

### 生產環境

```bash
npm run build
# 輸出到 dist/ 目錄
# 配置 Nginx 進行路由
```

## 📊 效能優化

1. **程式碼分割**：使用 React.lazy() 進行路由層級分割
2. **共用元件快取**：避免重複打包共用資源
3. **API 快取**：使用 React Query 進行智能快取
4. **圖片優化**：使用適當的圖片格式和大小

## 🔒 安全考量

1. **權限驗證**：Dashboard 路由需要驗證治療師身份
2. **Token 管理**：統一在 AuthContext 管理
3. **API 安全**：所有 API 請求都帶有認證 header
4. **資料驗證**：前端進行基本驗證，後端進行完整驗證
