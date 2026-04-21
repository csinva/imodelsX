PKG=imodelsx
cd ../$PKG
uv run pdoc --html . --output-dir ../docs --template-dir ../docs
cp -rf ../docs/$PKG/* ../docs/
rm -rf ../docs/$PKG
cd ../docs
rm -rf tests

# remove unnecessary files
# rm data.html
# rm embed.html
# rm linear.html
# rm index.html
# mv embgam.html index.html

# style the new file
uv run python style_docs.py

# render blog.md and insert it at the top of the main panel in index.html
uv run python - <<'PY'
import re
import markdown

with open('blog.md', 'r') as f:
    blog_html = markdown.markdown(f.read(), extensions=['tables'])

with open('index.html', 'r') as f:
    index = f.read()

index = index.replace(
    '<section id="section-intro">',
    f'<section id="blog">\n{blog_html}\n</section>\n<section id="section-intro">',
)

# strip content between the intro section opening and the models table header
index = re.sub(
    r'(<section id="section-intro">)\s*.*?(<p><strong>Explainable modeling/steering</strong></p>)',
    r'\1\n\2',
    index,
    count=1,
    flags=re.DOTALL,
)

# add a link to the top of the blog post as the first entry in the sidebar toc
index = index.replace(
    '<div class="toc">\n<ul>',
    '<div class="toc">\n<ul>\n'
    '<li><a href="#blog">Explaining text data by bridging interpretable models and LLMs</a></li>',
    1,
)

with open('index.html', 'w') as f:
    f.write(index)
PY