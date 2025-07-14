# CJK Fonts Directory

Place your CJK-compatible TrueType fonts (.ttf files) in this directory for PDF generation.

## Important Font Format Requirements

**Only .ttf fonts are supported!** 
- ✅ **TrueType fonts (.ttf)** - Fully supported
- ❌ **OpenType fonts (.otf)** - NOT supported by reportlab
- ❌ **TrueType Collection fonts (.ttc)** - NOT supported by reportlab

## Recommended Fonts (in order of preference)

### **Best CJK Fonts:**
1. **Arial Unicode MS** (Microsoft) - Universal Unicode font
   - File names: `ArialUnicodeMS.ttf` or `Arial Unicode MS.ttf`

2. **Source Han Sans** (Adobe) - Open-source CJK font family
   - Download: https://github.com/adobe-fonts/source-han-sans
   - **Important**: Download the .ttf version, not .otf
   - File names: `SourceHanSans-Regular.ttf` or `SourceHanSans.ttf`

3. **DejaVu Sans** - Good Unicode support
   - File name: `DejaVuSans.ttf`

### **Language-Specific Alternatives:**

#### For Chinese:
- **SimHei** (Chinese - Bold)
- **SimSun** (Chinese - Regular)

#### For Japanese:
- **MS Gothic** (Japanese)
- **Hiragino Sans** (Japanese)

#### For Korean:
- **Malgun Gothic** (Korean)
- **AppleGothic** (Korean) ✓ *Currently available*
- **Arial Unicode MS** (Universal)

## Important Notes

1. **Only .ttf files work** - .ttc and .otf files are not supported by reportlab
2. **OTF fonts will fail** - reportlab doesn't support PostScript outlines used in OTF fonts
3. **Fonts are not included** in the repository due to licensing
4. **Users must provide their own fonts** by copying them to this directory
5. **First available .ttf font** in the directory will be used
6. **No system fonts are used** - only fonts in this directory
7. **If no fonts available**, PDF generation will fall back to text files

## How to Add Fonts

1. Download or copy CJK-compatible .ttf font files
2. Place them in this `fonts/` directory
3. The script will automatically detect and use them

## Troubleshooting

If you're still seeing squares (■) in your PDF:
1. Make sure you have .ttf files (not .ttc) in this directory
2. Try different CJK fonts
3. Use text output (.txt) as an alternative - it always supports CJK characters
