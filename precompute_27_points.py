# Copyright 2026 cnomic-dev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

def generate_lookup_v01():
    """
    實作 STA v0.1 規範：預運算 27 個 S^3 語意錨點
    """
    lookup = {}
    states = [-1, 0, 1]
    
    print("🚀 正在生成 27 種語意狀態的 S^3 投影...")
    
    for I in states:
        for C in states:
            for O in states:
                # 根據 README 規範：phi(I, C, O) = (1, I, C, O) / norm
                v = np.array([1.0, float(I), float(C), float(O)])
                v_unit = v / np.linalg.norm(v)
                
                # 使用 tuple 作為 key 方便檢索
                lookup[(I, C, O)] = v_unit
                
    # 儲存為檔案供核心程式調用
    np.save('lookup_table.npy', lookup)
    print(f"✅ 已成功儲存 27 個錨點至 'lookup_table.npy'")
    
    # 驗證輸出
    test_key = (1, 1, 0) # 指令性, 正向
    print(f"範例檢索 {test_key} -> {lookup[test_key]}")

if __name__ == "__main__":
    generate_lookup_v01()
