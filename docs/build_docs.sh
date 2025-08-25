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