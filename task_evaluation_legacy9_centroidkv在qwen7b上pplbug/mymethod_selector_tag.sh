#!/usr/bin/env bash
# Canonical active-selector tag builder for experiment log/result filenames.
# Usage:
#   build_mymethod_selector_tag attention
#   build_mymethod_selector_tag keepkv SINK RECENT WINDOW BETA LOGC
#   build_mymethod_selector_tag d2o SINK RECENT LOGC RESET

_mymethod_tag_float() {
  printf '%.8g' "$1" | tr '.-' 'pm'
}

build_mymethod_selector_tag() {
  local selector="$1"
  case "${selector}" in
    attention)
      printf '%s' 'selector_attention'
      ;;
    keepkv)
      local sink="$2" recent="$3" window="$4" beta="$5" logc="$6"
      local beta_tag
      beta_tag="$(_mymethod_tag_float "${beta}")"
      printf 'selector_keepkv__sink%s__recent%s__window%s__beta%s__logc%s' \
        "${sink}" "${recent}" "${window}" "${beta_tag}" "${logc}"
      ;;
    d2o)
      local sink="$2" recent="$3" logc="$4" reset="$5"
      printf 'selector_d2o__sink%s__recent%s__logc%s__reset%s' \
        "${sink}" "${recent}" "${logc}" "${reset}"
      ;;
    *)
      echo "Unsupported MYMETHOD_SELECTOR=${selector}" >&2
      return 2
      ;;
  esac
}
