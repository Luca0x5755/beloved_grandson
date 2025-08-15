# 檔名: tts_service.py
# 說明: 此檔案整合了 TTS 模型引擎與 MinIO 上傳服務，提供一個完整的文字轉語音服務。

import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pyrubberband as pyrb
import soundfile as sf

# --- 模型相關導入 ---
import torch
from minio import Minio
from minio.error import S3Error
from mutagen import File
from opencc import OpenCC
from snac import SNAC
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- Hugging Face CLI 登入 ---
# 在 Docker 環境中，.env 的變數由 docker-compose 的 env_file 直接注入，
# 因此不需要使用 load_dotenv()。我們直接從 os.environ 讀取。
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
if HUGGING_FACE_TOKEN:
    print(">>> 偵測到 Hugging Face Token，正在執行 CLI 登入...", flush=True)
    try:
        # 執行登入指令。check=True 會在指令失敗時拋出錯誤。
        subprocess.run(
            ["huggingface-cli", "login", "--token", HUGGING_FACE_TOKEN],
            check=True,
            capture_output=True,  # 避免將 token 顯示在日誌中
            text=True,
        )
        print("✅ Hugging Face CLI 登入成功。", flush=True)
    except FileNotFoundError:
        print(
            "⚠️  警告: `huggingface-cli` 指令不存在。請確保 `huggingface-hub` 套件已安裝。",
            flush=True,
        )
    except subprocess.CalledProcessError as e:
        # 登入失敗，可能因為 token 無效
        print(
            f"⚠️  警告: Hugging Face CLI 登入失敗。Token 可能無效。錯誤: {e.stderr}",
            flush=True,
        )
else:
    print(">>> 未提供 Hugging Face Token，將以匿名方式進行操作。", flush=True)


# --- TTS 核心引擎 (原 optimized_inference_engine.py) ---
class OptimizedOrpheusTTS:
    """
    經過性能優化的 TTS 生成器，採用 4-bit 量化和 torch.compile 加速。
    """

    # 建構子不再需要 auth_token，因為已透過 CLI 全域登入
    def __init__(
        self,
        model_id: str,
        tokenizer_id: str,
        snac_id: str = "hubertsiuzdak/snac_24khz",
    ):
        print(">>> 正在初始化 Orpheus TTS 引擎 (已啟用性能優化)...", flush=True)

        compute_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

        print(f">>> 使用 4-bit 量化載入，計算精度為: {compute_dtype}", flush=True)

        # 不再需要傳遞 token 參數
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=quantization_config, device_map="auto"
        )

        print(">>> 正在使用 torch.compile() 編譯模型以獲取極致性能...", flush=True)
        try:
            self.model = torch.compile(
                self.model, mode="reduce-overhead", fullgraph=True
            )
            print("✅ 模型編譯成功。", flush=True)
        except Exception as e:
            print(f"⚠️ 模型編譯失敗，將使用未編譯版本。錯誤: {e}", flush=True)

        # 不再需要傳遞 token 參數
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self.snac_model = SNAC.from_pretrained(snac_id).cpu()
        self.cc = OpenCC("t2s.json")

        print("✅ Orpheus TTS 引擎已成功載入並優化。", flush=True)

    def _prepare_prompts_for_batch(
        self, prompts: List[str], voice: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        simplified_prompts = [self.cc.convert(p) for p in prompts]
        formatted_prompts = [f"{voice}: " + p for p in simplified_prompts]
        start_token = torch.tensor([[128259]], dtype=torch.long)
        end_tokens = torch.tensor([[128009, 128260]], dtype=torch.long)
        all_modified_input_ids = []
        for p in formatted_prompts:
            input_ids = self.tokenizer(p, return_tensors="pt").input_ids
            modified_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)
            all_modified_input_ids.append(modified_ids)
        max_length = max(ids.shape[1] for ids in all_modified_input_ids)
        all_padded_tensors, all_attention_masks = [], []
        padding_token_id = 128263
        for ids in all_modified_input_ids:
            padding_size = max_length - ids.shape[1]
            padded_tensor = torch.cat(
                [
                    torch.full((1, padding_size), padding_token_id, dtype=torch.long),
                    ids,
                ],
                dim=1,
            )
            attention_mask = torch.cat(
                [
                    torch.zeros((1, padding_size), dtype=torch.long),
                    torch.ones((1, ids.shape[1]), dtype=torch.long),
                ],
                dim=1,
            )
            all_padded_tensors.append(padded_tensor)
            all_attention_masks.append(attention_mask)
        return torch.cat(all_padded_tensors, dim=0), torch.cat(
            all_attention_masks, dim=0
        )

    def _decode_and_redistribute(
        self, generated_ids_batch: torch.Tensor
    ) -> List[torch.Tensor]:
        token_to_find, token_to_remove = 128257, 128258
        code_lists = []
        for row in generated_ids_batch:
            token_indices = (row == token_to_find).nonzero(as_tuple=True)[0]
            last_occurrence_idx = (
                token_indices[-1].item() if len(token_indices) > 0 else -1
            )
            cropped_tensor = row[last_occurrence_idx + 1 :]
            masked_row = cropped_tensor[cropped_tensor != token_to_remove]
            new_length = (masked_row.size(0) // 7) * 7
            trimmed_row = masked_row[:new_length]
            final_codes = [t.item() - 128266 for t in trimmed_row]
            code_lists.append(final_codes)
        output_waveforms = []
        for code_list in code_lists:
            if not code_list:
                output_waveforms.append(torch.tensor([]))
                continue
            layer_1, layer_2, layer_3 = [], [], []
            num_frames = len(code_list) // 7
            for i in range(num_frames):
                base = 7 * i
                layer_1.append(code_list[base])
                layer_2.append(code_list[base + 1] - 4096)
                layer_3.append(code_list[base + 2] - (2 * 4096))
                layer_3.append(code_list[base + 3] - (3 * 4096))
                layer_2.append(code_list[base + 4] - (4 * 4096))
                layer_3.append(code_list[base + 5] - (5 * 4096))
                layer_3.append(code_list[base + 6] - (6 * 4096))
            codes = [
                torch.tensor(layer, dtype=torch.int32).unsqueeze(0)
                for layer in [layer_1, layer_2, layer_3]
            ]
            with torch.no_grad():
                audio_hat = self.snac_model.decode(codes)
            output_waveforms.append(audio_hat)
        return output_waveforms

    def synthesize(
        self, prompt: str, voice: str, output_path: str, speed_rate: float = 1.0
    ):
        print(f"\n>>> 正在合成: '{prompt}'", flush=True)
        input_ids, attention_mask = self._prepare_prompts_for_batch([prompt], voice)
        input_ids, attention_mask = input_ids.to(self.model.device), attention_mask.to(
            self.model.device
        )

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1200,
                do_sample=True,
                temperature=0.4,
                top_p=0.65,
                repetition_penalty=1.5,
                eos_token_id=128258,
            )

        output_samples = self._decode_and_redistribute(generated_ids.to("cpu"))

        if output_samples and output_samples[0].numel() > 0:
            audio_numpy = output_samples[0].squeeze().numpy()
            if speed_rate != 1.0:
                print(f"   >>> 正在調整語速為: {speed_rate}x", flush=True)
                audio_numpy = pyrb.time_stretch(
                    y=audio_numpy, sr=24000, rate=speed_rate
                )

            sf.write(output_path, audio_numpy, 24000)
            print(f"✅ 音訊已成功儲存至: {output_path}", flush=True)
        else:
            print("⚠️ 警告：未生成有效音訊。", flush=True)


# --- 整合後的 TTS 服務 ---
class TTSService:
    def __init__(self):
        # MinIO 客戶端設定
        self.minio_client = Minio(
            os.environ.get("MINIO_ENDPOINT", "minio:9000"),
            access_key=os.environ.get("MINIO_ACCESS_KEY"),
            secret_key=os.environ.get("MINIO_SECRET_KEY"),
            secure=False,
        )
        self.bucket_name = os.environ.get("MINIO_BUCKET_NAME", "audio-bucket")
        self._ensure_bucket_exists()

        # TTS 引擎設定 (從 .env 讀取)
        MODEL_ID = os.environ.get("TTS_MODEL_ID")
        TOKENIZER_ID = os.environ.get("TTS_TOKENIZER_ID")
        self.default_voice = os.environ.get("TTS_DEFAULT_VOICE")

        # 檢查必要的設定是否存在，若否則拋出錯誤
        if not all([MODEL_ID, TOKENIZER_ID, self.default_voice]):
            raise ValueError(
                "TTS 服務缺少必要的環境變數設定。請在 .env 檔案中檢查 "
                "TTS_MODEL_ID, TTS_TOKENIZER_ID, 和 TTS_DEFAULT_VOICE 是否都已設定。"
            )

        self.tts_engine = OptimizedOrpheusTTS(
            model_id=MODEL_ID, tokenizer_id=TOKENIZER_ID
        )

    def _ensure_bucket_exists(self):
        """確保 MinIO bucket 存在。"""
        try:
            found = self.minio_client.bucket_exists(self.bucket_name)
            if not found:
                self.minio_client.make_bucket(self.bucket_name)
                print(f"Bucket '{self.bucket_name}' created.", flush=True)
        except S3Error as exc:
            print(f"Error checking or creating bucket: {exc}", flush=True)
            raise

    def synthesize_text(self, text: str) -> Tuple[str, int]:
        """
        使用 Orpheus TTS 引擎合成文字為音訊檔案，上傳到 MinIO，
        並返回物件名稱和音訊長度。
        語音和速度在內部固定。
        """
        # 根據使用者要求，將參數寫死
        speed_rate = 1.0
        voice = self.default_voice

        temp_audio_file = None
        try:
            # 1. 建立一個臨時檔案來儲存合成的音訊
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                temp_audio_file = tmp.name

            # 2. 合成音訊並儲存到臨時檔案
            self.tts_engine.synthesize(
                prompt=text,
                voice=voice,
                output_path=temp_audio_file,
                speed_rate=speed_rate,
            )

            # 3. 獲取音訊元數據並準備上傳
            if (
                not os.path.exists(temp_audio_file)
                or os.path.getsize(temp_audio_file) == 0
            ):
                raise ValueError("TTS 引擎未能產出有效的音訊檔案。")

            audio_data_len = os.path.getsize(temp_audio_file)

            metadata = {}
            duration_ms = 0
            try:
                # audio_file_info = File(temp_audio_file)
                # if audio_file_info and audio_file_info.info:
                #     duration_ms = int(audio_file_info.info.length * 1000)
                with sf.SoundFile(temp_audio_file) as f:
                    duration_ms = int(len(f) / f.samplerate * 1000)
                    metadata["duration-ms"] = str(duration_ms)
            except Exception as e:
                print(f"無法使用 mutagen 讀取音訊元數據: {e}", flush=True)

            # 4. 上傳到 MinIO
            object_name = f"{uuid.uuid4()}.m4a"
            print(
                f"Uploading {object_name} to bucket {self.bucket_name}...", flush=True
            )

            with open(temp_audio_file, "rb") as audio_file_data:
                self.minio_client.put_object(
                    self.bucket_name,
                    object_name,
                    audio_file_data,
                    length=audio_data_len,
                    content_type="audio/m4a",
                    metadata=metadata,
                )

            print(f"成功上傳 {object_name}, 長度: {duration_ms}ms.", flush=True)
            return object_name, duration_ms

        except Exception as e:
            print(f"在 TTS 合成或上傳過程中發生錯誤: {e}", flush=True)
            raise
        finally:
            # 5. 清理臨時檔案
            if temp_audio_file and os.path.exists(temp_audio_file):
                os.remove(temp_audio_file)
                print(f"已刪除臨時檔案: {temp_audio_file}", flush=True)


# --- 單例實例與工廠模式 ---
_tts_service_instance: Optional[TTSService] = None


def get_tts_service() -> TTSService:
    """工廠函式，用於獲取 TTSService 的單例。"""
    global _tts_service_instance
    if _tts_service_instance is None:
        print("首次初始化 TTSService...", flush=True)
        _tts_service_instance = TTSService()
    return _tts_service_instance


# --- 更新後的測試區塊 ---
if __name__ == "__main__":
    # 此區塊展示如何使用整合後的 TTSService。
    # 執行此測試:
    # 1. 確保您有一個包含 MinIO 憑證和 HUGGING_FACE_TOKEN 的 .env 檔案。
    # 2. 啟動 ai-worker 容器:
    #    docker-compose -f docker-compose.dev.yml up -d --build ai-worker
    # 3. 在容器內執行此腳本:
    #    docker-compose -f docker-compose.dev.yml exec ai-worker python worker/tts_app/tts_service.py

    print("\n--- 開始 TTSService 整合測試 ---")
    try:
        tts_service = get_tts_service()

        test_text = "大家好，這是一個測試，看看整合後的文字轉語音服務是否正常運作。"
        print(f"正在合成測試文字: '{test_text}'")

        object_name, duration_ms = tts_service.synthesize_text(test_text)

        print("\n--- 測試結果 ---")
        print(f"合成的音訊已上傳為: {object_name}")
        print(f"音訊長度: {duration_ms} 毫秒")
        print("測試成功結束。")
        print(
            f"\n若要驗證，請在您的 MinIO 實例中 (例如：http://localhost:9001) 檢查名為 '{tts_service.bucket_name}' 的 bucket。"
        )

    except Exception as e:
        import traceback

        print(f"\n測試過程中發生錯誤: {e}")
        traceback.print_exc()
        traceback.print_exc()
