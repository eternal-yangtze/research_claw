"""Local invited accounts. Passwords are salted; legacy shared login is opt-in."""
import hashlib
import secrets

from fastapi import HTTPException
from pydantic import BaseModel, Field, field_validator

USER_ID_PATTERN = r'^[a-zA-Z0-9_@.-]{1,64}$'


class UserIdentity(BaseModel):
    user_id: str = Field(pattern=USER_ID_PATTERN)

    @field_validator('user_id')
    @classmethod
    def safe_identity(cls, value):
        if value in {'.', '..'}:
            raise ValueError('用户 ID 不能是 . 或 ..')
        return value


class Registration(UserIdentity):
    password: str = Field(min_length=10, max_length=256)
    invite_code: str = Field(max_length=256)


class Accounts:
    def __init__(self, store):
        self.store = store
        store.database.migrate('014-local-accounts', '''CREATE TABLE web_accounts (
            user_id TEXT PRIMARY KEY, salt TEXT NOT NULL, password_hash TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now());''')

    @staticmethod
    def derive(password, salt):
        return hashlib.pbkdf2_hmac('sha256', password.encode(), bytes.fromhex(salt), 260000).hex()

    def register(self, body, invite):
        if not secrets.compare_digest(body.invite_code.encode(), invite.encode()):
            raise HTTPException(403, '邀请码不正确；使用本机 .env.web 中的访问密码作为邀请码')
        salt = secrets.token_hex(24)
        with self.store.connect() as db:
            # An existing historical ID must not be silently claimed by registration.
            if db.execute('SELECT 1 FROM sessions WHERE owner=? LIMIT 1', (body.user_id,)).fetchone():
                raise HTTPException(409, '此 ID 有历史会话，请使用历史账号认领入口')
            result = db.execute('INSERT INTO web_accounts VALUES(?,?,?,now()) ON CONFLICT DO NOTHING',
                (body.user_id, salt, self.derive(body.password, salt)))
            if not result.rowcount:
                raise HTTPException(409, '用户 ID 已注册')

    def verify(self, identity, password):
        with self.store.connect() as db:
            row = db.execute('SELECT * FROM web_accounts WHERE user_id=?', (identity,)).fetchone()
        return None if not row else secrets.compare_digest(self.derive(password, row['salt']), row['password_hash'])

    def claim(self, body, invite):
        if not secrets.compare_digest(body.invite_code.encode(), invite.encode()):
            raise HTTPException(403, '历史账号认领需要本机邀请码')
        salt = secrets.token_hex(24)
        with self.store.connect() as db:
            if not db.execute('SELECT 1 FROM sessions WHERE owner=? LIMIT 1', (body.user_id,)).fetchone():
                raise HTTPException(404, '没有此 ID 的历史会话，请注册新账号')
            row = db.execute('INSERT INTO web_accounts VALUES(?,?,?,now()) ON CONFLICT DO NOTHING',
                (body.user_id, salt, self.derive(body.password, salt)))
            if not row.rowcount:
                raise HTTPException(409, '账号已认领，请直接登录')
