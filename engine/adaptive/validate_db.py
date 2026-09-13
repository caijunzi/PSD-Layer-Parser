"""
校验数据库完整性（CI/测试前执行）
"""
import sqlite3
import sys

def validate_db(db_path: str) -> bool:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    checks = []
    
    # 检查 1: 类目数量 ≥ 20
    cur.execute("SELECT COUNT(*) FROM categories")
    n_cat = cur.fetchone()[0]
    checks.append(("类目数量 ≥ 20", n_cat >= 20, n_cat))
    
    # 检查 2: 提示词数量 ≥ 40
    cur.execute("SELECT COUNT(*) FROM category_prompts WHERE source='seed'")
    n_prompt = cur.fetchone()[0]
    checks.append(("种子提示词 ≥ 40", n_prompt >= 40, n_prompt))
    
    # 检查 3: 材质亲和度 ≥ 100
    cur.execute("SELECT COUNT(*) FROM category_material_affinity")
    n_affinity = cur.fetchone()[0]
    checks.append(("材质亲和度 ≥ 100", n_affinity >= 100, n_affinity))
    
    # 检查 4: 激活版本存在
    cur.execute("SELECT version_id FROM active_version WHERE id=1")
    row = cur.fetchone()
    checks.append(("激活版本存在", row is not None, row[0] if row else None))
    
    # 检查 5: 所有类目至少有 1 个提示词
    cur.execute("""
        SELECT c.id FROM categories c
        LEFT JOIN category_prompts cp ON c.id = cp.category_id
        GROUP BY c.id HAVING COUNT(cp.id) = 0
    """)
    orphan_cats = cur.fetchall()
    checks.append(("无孤儿类目（无提示词）", len(orphan_cats) == 0, orphan_cats))
    
    # 检查 6: 核心类目（印章/题跋/修复痕迹）confidence ≥ 0.85
    cur.execute("""
        SELECT id, confidence FROM categories
        WHERE id IN ('seal', 'calligraphy', 'repair_marks')
          AND confidence < 0.85
    """)
    low_conf = cur.fetchall()
    checks.append(("核心类目置信度 ≥ 0.85", len(low_conf) == 0, low_conf))
    
    conn.close()
    
    # 打印结果
    all_pass = True
    for desc, passed, detail in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {desc}: {detail}")
        if not passed:
            all_pass = False
    
    return all_pass

if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "webui/data/adaptive_semantics.db"
    passed = validate_db(db_path)
    sys.exit(0 if passed else 1)
