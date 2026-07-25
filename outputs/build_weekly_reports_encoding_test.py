from pathlib import Path
p = Path(r'F:\Internship\Bytedance\docs') / '第三周总结_test.txt'
p.write_text('第三周总结 测试', encoding='utf-8')
print(p)
