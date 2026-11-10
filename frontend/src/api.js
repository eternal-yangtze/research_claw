export class ApiError extends Error {
  constructor(status, detail) {
    const message = Array.isArray(detail) ? detail.map(item => {
      const field = item.loc?.at(-1);
      if (field === 'user_id') return '用户 ID 须为 1–64 位英文字母、数字、下划线、连字符、@ 或点号，不能是单独的 . 或 ..';
      if (field === 'password') return '个人密码须为 10–256 位，支持大小写字母、数字和特殊符号';
      return `${field || '输入'}：${item.msg || '格式不正确'}`;
    }).join('；') : typeof detail === 'string' ? detail : JSON.stringify(detail);
    super(message); this.status = status;
  }
}
export async function api(path, options = {}) {
  const response = await fetch('/api/v1' + path, { credentials: 'include', ...options });
  const value = await response.json().catch(() => ({}));
  if (!response.ok) throw new ApiError(response.status, value.detail || '请求未成功，请稍后重试');
  return value;
}
export const post = (path, body = {}) => api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
export function uploadFile(file, query, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/v1/knowledge/upload?' + new URLSearchParams({ ...query, stage: 'true', suffix: 'pdf', filename: file.name }));
    xhr.withCredentials = true;
    xhr.upload.onprogress = event => { if (event.lengthComputable) onProgress(Math.round(100 * event.loaded / event.total)); };
    xhr.onload = () => {
      let body = {}; try { body = JSON.parse(xhr.responseText); } catch { /* handled below */ }
      xhr.status < 300 ? resolve(body) : reject(new ApiError(xhr.status, body.detail || '上传失败'));
    };
    xhr.onerror = () => reject(new Error('上传连接中断；请检查网络后重试'));
    xhr.send(file);
  });
}
