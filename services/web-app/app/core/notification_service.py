# services/web-app/app/core/notification_service.py
import pika # RabbitMQ 的 Python 客戶端函式庫
import os # 用於讀取環境變數
import json # 用於解析 JSON 格式的訊息
import threading # 用於在背景執行緒中運行監聽器，避免阻塞主程式
import time # 用於在重試連線時暫停
import logging # 用於記錄錯誤日誌
from functools import partial # 用於包裝回呼函式，傳遞額外參數
from app.extensions import socketio # 導入 Socket.IO 擴展，用於 WebSocket 通訊
from app.core.chat_repository import ChatRepository # 導入聊天記錄的資料庫操作模組 (此處未實際使用)

def message_callback(ch, method, properties, body, app):
    """
    處理從 RabbitMQ 收到的訊息。此函式會在 Flask 的應用程式上下文中執行。
    ch: channel 物件
    method: 訊息傳遞的元數據
    properties: 訊息的屬性
    body: 訊息的內容 (bytes)
    app: Flask 應用程式實例
    """
    # 確保在 Flask 的應用程式上下文 (app_context) 中執行，以便能使用 Flask 的擴展功能
    with app.app_context():
        print(f" [x] 收到通知: {body.decode()}", flush=True)
        try:
            # 將收到的 bytes 格式訊息解碼為 JSON 物件
            message = json.loads(body)
            print(message, '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

            patient_id = message.get("patient_id") # 獲取使用者 ID
            ai_response = message.get("ai_response") # 獲取 AI 回應內容

            # 檢查必要欄位是否存在
            if not patient_id or not ai_response:
                raise ValueError("通知訊息缺少 'patient_id' 或 'ai_response' 欄位。")

            # --- 1. 透過 WebSocket 將通知推播到 Web 前端 ---
            print(f" [>] 正在透過 WebSocket 發送通知: {message}", flush=True)
            # 使用 socketio.emit 向指定房間(room)發送 'notification' 事件
            # 房間名稱設定為 patient_id，確保只有該使用者會收到
            socketio.emit('notification', message, room=str(patient_id))

            # --- 2. 將通知推播回 LINE ---
            print(f" [>] 正在將通知推播到 LINE 給使用者 {patient_id}", flush=True)
            from .line_service import get_line_service
            line_service = get_line_service()

            # 根據通知內容決定要傳送文字還是音訊
            response_audio_url = message.get("response_audio_url")
            if response_audio_url:
                # 如果有音訊 URL，則先傳送一段引導文字，再傳送音訊
                line_service.push_text_message(user_id=patient_id, text=ai_response)

                # 直接從訊息中獲取由 ai-worker 計算好的音訊時長
                duration_ms = message.get("audio_duration_ms", 60000) # 若無提供，預設為 60 秒

                line_service.push_audio_message(
                    user_id=patient_id,
                    object_name=response_audio_url,
                    duration=duration_ms
                )
            else:
                # 如果沒有音訊 URL，只傳送文字回應
                line_service.push_text_message(user_id=patient_id, text=ai_response)

        except (json.JSONDecodeError, ValueError) as e:
            # 處理 JSON 解析錯誤或數值錯誤
            logging.error(f"無效的訊息格式或 JSON 解碼失敗: {e}")
        except Exception as e:
            # 處理其他所有未預期的錯誤
            logging.error(f" [!] 處理訊息時發生錯誤: {e}", exc_info=True)

        # 向 RabbitMQ 發送確認信號 (acknowledgment)，表示訊息已成功處理
        # 這樣 RabbitMQ 才會將該訊息從佇列中移除
        ch.basic_ack(delivery_tag=method.delivery_tag)

def start_notification_listener(app):
    """
    在一個背景執行緒中啟動 RabbitMQ 監聽器。
    """
    # 創建一個新的執行緒，目標是執行 listen_for_notifications 函式
    thread = threading.Thread(target=listen_for_notifications, args=(app,))
    # 將執行緒設置為守護執行緒 (daemon)，這樣主程式結束時該執行緒也會跟著結束
    thread.daemon = True
    # 啟動執行緒
    thread.start()

def listen_for_notifications(app):
    """
    連接到 RabbitMQ 並監聽來自 ai-worker 的通知。
    這個函式會在一個無限循環中運行，以確保連線的穩定性。
    """
    # 從環境變數讀取 RabbitMQ 的主機和佇列名稱，若無則使用預設值
    rabbitmq_host = os.environ.get("RABBITMQ_HOST", "rabbitmq")
    notification_queue = os.environ.get("RABBITMQ_NOTIFICATION_QUEUE", "notifications_queue")

    # 無限循環，用於自動重連
    while True:
        try:
            # 建立與 RabbitMQ 的連線
            connection = pika.BlockingConnection(pika.ConnectionParameters(host=rabbitmq_host))
            channel = connection.channel()
            # 宣告一個持久化的 (durable) 佇列，確保 RabbitMQ 重啟後訊息不會遺失
            channel.queue_declare(queue=notification_queue, durable=True)
            print(' [*] 通知監聽器已啟動，等待訊息中。按 CTRL+C 離開')

            # # 使用 functools.partial 將 app 參數預先綁定到 message_callback 函式
            # # 因為 pika 的回呼函式簽名不包含 app 參數，需要透過這種方式傳遞
            on_message_callback = partial(message_callback, app=app)

            # # 設定消費者 (consumer)，指定從哪個佇列接收訊息以及處理訊息的回呼函式
            channel.basic_consume(queue=notification_queue, on_message_callback=on_message_callback)
            # # 開始監聽，此為一個阻塞操作，會一直等待訊息進來

            channel.start_consuming()

        except pika.exceptions.AMQPConnectionError as e:
            # 如果連線失敗，印出錯誤訊息並等待 5 秒後重試
            print(f"無法連線到 RabbitMQ: {e}。5 秒後重試...")
            time.sleep(5)
        except Exception as e:
            # 捕獲其他所有異常，防止監聽器崩潰
            print(f"監聽器發生未預期錯誤: {e}。正在重新啟動...")
            time.sleep(5)
