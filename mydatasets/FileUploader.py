import sqlite3
import os
from datetime import datetime
from openai import OpenAI
from mydatasets.BaseDataset import BaseDataset
from pathlib import Path
import toml


class FileUploader:
    def __init__(self, dataset: BaseDataset) -> None:
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.sqlite_path = dataset.sqlite_path
        self.dataset_name = dataset.dataset_name
        self.create_files_table()
        self.document_path = dataset.document_path

    def create_files_table(self):
        """创建文件记录表"""
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS uploaded_files (
                    id TEXT PRIMARY KEY,           -- OpenAI 文件 ID
                    object TEXT,
                    bytes INTEGER,
                    created_at INTEGER,
                    filename TEXT,
                    purpose TEXT,
                    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- 本地上传时间
                    dataset_name TEXT,
                    raw_filename TEXT
                )
            """
            )
            conn.commit()

    def upload_file_and_store(self, file_path: str, purpose: str):
        """
        上传文件到 OpenAI，并将文件元信息存储到 SQLite 数据库

        Args:
            file_path (str): 要上传的本地文件路径
            purpose (str): 文件用途，如 'fine-tune', 'assistants', 'batch', 'evals', 'user_data', 'vision'

        Returns:
            dict: OpenAI 返回的文件对象
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件未找到: {file_path}")

        try:
            filename = Path(file_path).name
            print(f"正在上传文件: {file_path} (用途: {purpose})...")

            # 打开文件并上传
            with open(file_path, "rb") as f:
                response = self.client.files.create(file=f, purpose=purpose)  # type: ignore

            # 将响应转换为字典
            file_data = (
                response.model_dump()
            )  # 使用 model_dump() 获取字典（适用于 OpenAI v1.x+）

            # 提取字段
            file_record = (
                file_data["id"],
                file_data["object"],
                file_data["bytes"],
                file_data["created_at"],
                file_data["filename"],
                file_data["purpose"],
                self.dataset_name,
                filename,
            )

            # 插入数据库
            with sqlite3.connect(self.sqlite_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO uploaded_files
                    (id, object, bytes, created_at, filename, purpose, dataset_name, raw_filename)
                    VALUES (?, ?, ?, ?, ?, ?, ?,?)
                """,
                    file_record,
                )
                conn.commit()

            print(
                f"文件上传成功！ID: {file_data['id']}, 文件名: {file_data['filename']}"
            )
            return file_data

        except Exception as e:
            print(f"文件上传失败: {str(e)}")
            return

    def get_fileID_by_name(self, filename: str) -> str | None:
        """
        根据文件名和用途查询文件 ID

        Args:
            filename (str): 上传时的原始文件名（注意：是上传后 OpenAI 返回的 filename）
        Returns:
            str or None: 匹配的文件 ID，如果没有找到则返回 None
        """
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM uploaded_files WHERE dataset_name = ? AND raw_filename = ?",
                (self.dataset_name, filename),
            )
            row = cursor.fetchone()
        file_id = row["id"] if row else None
        if file_id:
            return file_id
        else:
            file_object = self.upload_file_and_store(
                os.path.join(self.document_path, filename), "assistants"
            )
            return file_object["id"] if file_object else None

    def get_fileID_by_path(self, filepath: str) -> str | None:
        filename = Path(filepath).name
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM uploaded_files WHERE dataset_name = ? AND raw_filename = ?",
                (self.dataset_name, filename),
            )
            row = cursor.fetchone()
        file_id = row["id"] if row else None
        if file_id:
            return file_id
        else:
            file_object = self.upload_file_and_store(filepath, "assistants")
            return file_object["id"] if file_object else None
