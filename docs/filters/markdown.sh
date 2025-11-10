#!/bin/bash

# Unused
readonly regex_match_math_double='^\$\$$'
readonly regex_substitute_math_double='\\f$$'

readonly regex_match_math_split='\$(.)\$'
readonly regex_substitute_math_split='$ \1 $'

readonly regex_match_math='(([^$])\$|^\$([^$]))'
readonly regex_substitute_math='\2\\f$\3'

readonly regex_match_slash_curly_math='([^\])\\(\{|\})'
readonly regex_substitute_slash_curly_math='\1\\\\\2'

readonly regex_match_code_block='^```'
readonly regex_substitute_code_block='\d30'

readonly regex_match_tab_tag='\[\/\/\]: # "@tab"'
readonly regex_substitute_tab_tag='\d16'

readonly regex_match_end_tab_tag='\[\/\/\]: # "@end-tab"'
readonly regex_substitute_end_tab_tag='\d17'

readonly regex_match_tabbed='\[\/\/\]: # "@tabbed"'
readonly regex_substitute_tabbed='<div class="tabbed"><ul>'

readonly regex_match_tabbed_end='\[\/\/\]: # "@end-tabbed"'
readonly regex_substitute_tabbed_end='<\/ul><\/div>'

readonly regex_match_tab='\d16[^#]+#+ *([^\n]*)\n([^\d17]+)\d17'
readonly regex_substitute_tab='<li><b id="\1" class="tab-title">\1<\/b>\n\2\n<\/li>'

readonly regex_match_mermaid='\d30mermaid\n([^\d30]*)\n\d30'
readonly regex_substitute_mermaid="<pre class='mermaid'>\n\1<\/pre>"

readonly regex_match_code='\d30(\w+)\n([^\d30]*)\n\d30'
readonly regex_substitute_code="<pre><code class='fragment language-\1'>\2<\/code><\/pre>"

readonly regex_match_root_link='\]\(\/[^)]+\/([^)]+)\)'
readonly regex_substitute_root_link='](\1)'

readonly regex_title_logo='<img alt="Icon" src="docs\/_static\/logo.svg" align="left" width="35" height="35">'


cat "${1}" \
| sed -E \
    -e "s/$regex_match_math_split/$regex_substitute_math_split/g" \
    -e "s/$regex_match_slash_curly_math/$regex_substitute_slash_curly_math/g" \
    -e "s/$regex_match_math/$regex_substitute_math/g" \
    -e "s/$regex_title_logo//g" \
    -e "s/$regex_match_code_block/$regex_substitute_code_block/g" \
    -e "s/$regex_match_tab_tag/$regex_substitute_tab_tag/g" \
    -e "s/$regex_match_end_tab_tag/$regex_substitute_end_tab_tag/g" \
    -e "s/$regex_match_tabbed/$regex_substitute_tabbed/g" \
    -e "s/$regex_match_tabbed_end/$regex_substitute_tabbed_end/g" \
| sed -E -z  \
    -e "s/$regex_match_mermaid/$regex_substitute_mermaid/g" \
    -e "s/$regex_match_code/$regex_substitute_code/g" \
    -e "s/$regex_match_root_link/$regex_substitute_root_link/g" \
    -e "s/$regex_match_tab/$regex_substitute_tab/g"
