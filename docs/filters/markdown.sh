#!/bin/bash

readonly regex_match_slash_curly_math='([^\])\\(\{|\})'
readonly regex_substitute_slash_curly_math='\1\\\\\2'

# End of a code block. Identified by the fact that it is just ```, with no language specifier
readonly regex_match_code_block_end='^```$'
readonly regex_substitute_code_block_end='\d31'

# Start of a code block. Identified by ``` at the beginning of a line with a language specifier
readonly regex_match_code_block_start='^```'
readonly regex_substitute_code_block_start='\d30'
# Tab tags
readonly regex_match_tab_tag='\[\/\/\]: # "@tab"'
readonly regex_substitute_tab_tag='\d16'

# End tab tag
readonly regex_match_end_tab_tag='\[\/\/\]: # "@end-tab"'
readonly regex_substitute_end_tab_tag='\d17'

# Tabbed section tags
readonly regex_match_tabbed='\[\/\/\]: # "@tabbed"'
readonly regex_substitute_tabbed='<div class="tabbed"><ul>'

# End tabbed section tags
readonly regex_match_tabbed_end='\[\/\/\]: # "@end-tabbed"'
readonly regex_substitute_tabbed_end='<\/ul><\/div>'

# Individual tabs. The tab title is captured in group 1, which corresponds to a markdown header, the tab content in group 2
readonly regex_match_tab='\d16[^#]+#+ *([^\n]*)\n([^\d17]+)\d17'
readonly regex_substitute_tab='<li><b id="\1" class="tab-title">\1<\/b>\n\2\n<\/li>'

# Remove dollar signs in code blocks
readonly regex_match_math_in_code='\d30([^\d30\d31]*)\$([^\d30\d31]*)\n\d31'
readonly regex_substitute_math_in_code='\d30\1\d20\2\n\d31'

# Math expressions enclosed in double dollar signs, which must be at the start of a line
readonly regex_match_math_double='^\$\$$'
readonly regex_substitute_math_double='\\f$'

# Math single character expressions enclosed in single dollar signs, ensuring that there is a space before and after
readonly regex_match_math_split='\$(.)\$'
readonly regex_substitute_math_split='$ \1 $'

# Math single character expressions enclosed in single dollar signs, ensuring that there is not a backslash before the first dollar sign and not a dollar sign after the second
readonly regex_match_math='([^\d20$\])\$([^$\d20])'
readonly regex_substitute_math='\1\\f$\2'

# Math single character expressions enclosed in single dollar signs, only at the start of a line
readonly regex_match_math_start='^\$([^$\d20])'
readonly regex_substitute_math_start='\\f$\1'

# Mermaid diagrams
readonly regex_match_mermaid='\d30mermaid\n([^\d30\d31]*)\n\d31'
readonly regex_substitute_mermaid="<pre class='mermaid'>\n\1<\/pre>"

# Code blocks
readonly regex_match_code='\d30(\w+)[^\n]*\n([^\d30\d31]*)\n\d31'
readonly regex_substitute_code="<pre><code class='fragment language-\1'>\2<\/code><\/pre>"

# Links to root-relative paths
readonly regex_match_root_link='\]\(\/[^)]+\/([^)]+)\)'
readonly regex_substitute_root_link='](\1)'

# Title logo
readonly regex_title_logo='<img alt="Icon" src="docs\/_static\/logo.svg" align="left" width="35" height="35">'

# Restore dollar signs in code blocks
readonly regex_match_dollar='\d20'
readonly regex_substitute_dollar='$'

# Read the input file
# 1. Remove title logo
# 2. Replace code block delimiters with placeholders
# 3. Replace tab tags with placeholders
# 4. Replace tabbed section tags with HTML placeholders
# 5. Handle math in code blocks
# 6. Handle math expressions
# 7. Handle mermaid diagrams
# 8. Handle code blocks
# 9. Handle root-relative links
# 10. Handle tabs
# 11. Restore dollar signs in code blocks
# Note that we use -z with sed when processing multi-line patterns
cat "${1}" \
| sed -E \
    -e "s/$regex_title_logo//g" \
    -e "s/$regex_match_code_block_end/$regex_substitute_code_block_end/g" \
    -e "s/$regex_match_code_block_start/$regex_substitute_code_block_start/g" \
    -e "s/$regex_match_tab_tag/$regex_substitute_tab_tag/g" \
    -e "s/$regex_match_end_tab_tag/$regex_substitute_end_tab_tag/g" \
    -e "s/$regex_match_tabbed/$regex_substitute_tabbed/g" \
    -e "s/$regex_match_tabbed_end/$regex_substitute_tabbed_end/g" \
| sed -E -z \
    -e "s/$regex_match_math_in_code/$regex_substitute_math_in_code/g " \
| sed -E \
    -e "s/$regex_match_math_double/$regex_substitute_math_double/g" \
    -e "s/$regex_match_math_split/$regex_substitute_math_split/g" \
    -e "s/$regex_match_math/$regex_substitute_math/g" \
    -e "s/$regex_match_math_start/$regex_substitute_math_start/g" \
| sed -E -z  \
    -e "s/$regex_match_mermaid/$regex_substitute_mermaid/g" \
    -e "s/$regex_match_code/$regex_substitute_code/g" \
    -e "s/$regex_match_root_link/$regex_substitute_root_link/g" \
    -e "s/$regex_match_tab/$regex_substitute_tab/g" \
    -e "s/$regex_match_dollar/$regex_substitute_dollar/g" \
