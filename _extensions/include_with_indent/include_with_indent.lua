-- include_with_indent.lua
-- A Quarto extension that includes content from files

function include_with_indent(args, kwargs, meta, raw_args, context)
  quarto.log.output("include_with_indent called with path: " .. tostring(args[1]))
  
  -- Log kwargs safely
  local kwargs_str = "{"
  if kwargs then
    for k, v in pairs(kwargs) do
      kwargs_str = kwargs_str .. k .. "=" .. tostring(v) .. ", "
    end
  end
  kwargs_str = kwargs_str .. "}"
  quarto.log.output("kwargs: " .. kwargs_str)
  quarto.log.output("context: " .. tostring(context))
  
  -- Get the file path from the first argument
  local file_path = args[1]
  
  if not file_path then
    quarto.log.warning("Missing file path for include_with_indent shortcode")
    return pandoc.Null()
  end
  
  -- Resolve the file path
  local paths_to_try = resolve_file_path(file_path, meta)
  
  -- Try to read the file content
  local content = read_file(paths_to_try)
  if not content then
    quarto.log.warning("Failed to read file from any of the tried paths")
    quarto.log.output("Tried paths: ")
    for _, path in ipairs(paths_to_try) do
      quarto.log.output("  - " .. path)
    end
    return pandoc.Null()
  end
  
  -- Strip YAML frontmatter if present
  local stripped_content = strip_yaml_frontmatter(content)
  
  -- Parse the content with Pandoc based on the context
  return parse_content(stripped_content, context)
end

-- Helper function to get directory part of a path
function get_directory(path)
  return path:match("(.*[/\\])")
end

-- Helper function to resolve file path
function resolve_file_path(file_path, meta)
  local paths_to_try = {}
  
  -- Check if it's an absolute path (only Windows drive letter paths are considered absolute)
  local is_absolute = is_absolute_path(file_path)
  if is_absolute then
    table.insert(paths_to_try, file_path)
    quarto.log.output("Added absolute path: " .. file_path)
    return paths_to_try
  end
  
  -- Original path as-is
  table.insert(paths_to_try, file_path)
  quarto.log.output("Added original path: " .. file_path)
  
  -- Get the project directory
  local project_dir = quarto.project.directory and quarto.project.directory()
  if project_dir then
    quarto.log.output("Project directory: " .. project_dir)
  else
    quarto.log.output("No project directory found")
  end
  
  -- Get the document directory
  local doc_dir = meta and meta.documentdir and pandoc.utils.stringify(meta.documentdir)
  if doc_dir then
    quarto.log.output("Document directory: " .. doc_dir)
  else
    quarto.log.output("No document directory found in meta")
  end
  
  -- Get current working directory
  local cwd = io.popen("pwd"):read("*l")
  if cwd then
    quarto.log.output("Current working directory: " .. cwd)
  else
    quarto.log.output("Could not determine current working directory")
    cwd = "."  -- Fallback to current directory
  end
  
  -- Path relative to current working directory if path starts with /
  if string.sub(file_path, 1, 1) == "/" then
    local relative_path = string.sub(file_path, 2)  -- Remove leading slash
    local cwd_relative_path = pandoc.path.join({cwd, relative_path})
    table.insert(paths_to_try, cwd_relative_path)
    quarto.log.output("Added cwd-relative path: " .. cwd_relative_path)
    
    -- Also try with project directory
    if project_dir then
      local project_relative_path = pandoc.path.join({project_dir, relative_path})
      table.insert(paths_to_try, project_relative_path)
      quarto.log.output("Added project-relative path: " .. project_relative_path)
    end
    
    -- Also try with document directory
    if doc_dir then
      local doc_relative_path = pandoc.path.join({doc_dir, relative_path})
      table.insert(paths_to_try, doc_relative_path)
      quarto.log.output("Added doc-relative path with leading slash: " .. doc_relative_path)
    end
  end
  
  -- Always try relative to document directory
  if doc_dir then
    local direct_join_path = pandoc.path.join({doc_dir, file_path})
    table.insert(paths_to_try, direct_join_path)
    quarto.log.output("Added direct doc join path: " .. direct_join_path)
    
    -- Special case: try going up one directory from the document
    local parent_dir = get_directory(doc_dir)
    if parent_dir then
      local parent_join_path = pandoc.path.join({parent_dir, file_path})
      table.insert(paths_to_try, parent_join_path)
      quarto.log.output("Added parent directory join path: " .. parent_join_path)
    end
  end
  
  -- Try relative to project root regardless of leading slash
  if project_dir then
    local project_join_path = pandoc.path.join({project_dir, file_path})
    if not has_path(paths_to_try, project_join_path) then
      table.insert(paths_to_try, project_join_path)
      quarto.log.output("Added project join path: " .. project_join_path)
    end
  end
  
  return paths_to_try
end

-- Helper function to check if a path is already in the list
function has_path(paths, path)
  for _, p in ipairs(paths) do
    if p == path then
      return true
    end
  end
  return false
end

-- Helper function to check if path is absolute
function is_absolute_path(path)
  -- Only consider Windows drive letter paths as absolute (e.g., C:\)
  -- Paths starting with / are no longer considered absolute
  return path and string.find(path, "^%a:[\\/]") ~= nil
end

-- Helper function to read file from a list of possible paths
function read_file(paths_to_try)
  for _, path in ipairs(paths_to_try) do
    quarto.log.output("Trying path: " .. path)
    local success, content = pcall(function()
      local file = io.open(path, "r")
      if not file then
        return nil
      end
      local content = file:read("*all")
      quarto.log.output("File read successfully from: " .. path)
      file:close()
      return content
    end)
    
    if success and content then
      -- Store the include directory for path resolution later
      if quarto.project.directory then
        local file_metadata = quarto.project.current_file and quarto.project.current_file()
        if file_metadata then
          file_metadata.include_directory = get_directory(path)
          quarto.log.output("Set include_directory to: " .. get_directory(path))
        else
          quarto.log.output("No file metadata available")
        end
      else
        quarto.log.output("quarto.project.directory not available")
      end
      
      return content
    end
  end
  
  return nil
end

-- Helper function to strip YAML frontmatter
function strip_yaml_frontmatter(content)
  if string.sub(content, 1, 3) == "---" then
    quarto.log.output("YAML frontmatter detected, stripping it")
    -- Look for the ending delimiter which could be either "---" or "..."
    local _, end_pos = string.find(content, "\n%-%-%-\n", 4)
    if not end_pos then
      _, end_pos = string.find(content, "\n%.%.%.\n", 4)
    end
    if end_pos then
      quarto.log.output("Stripped YAML frontmatter")
      return string.sub(content, end_pos + 1)
    else
      quarto.log.output("Could not find end of YAML frontmatter, using original content")
    end
  end
  return content
end

-- Helper function to parse content based on context
function parse_content(content, context)
  if context == "block" then
    -- For block context, we process as Markdown and return blocks
    quarto.log.output("Processing as block context")
    local doc = pandoc.read(content)
    quarto.log.output("Number of blocks: " .. tostring(#doc.blocks))
    return doc.blocks
  else
    -- For inline context, try to get the first paragraph's content
    quarto.log.output("Processing as inline context")
    local doc = pandoc.read(content)
    if #doc.blocks > 0 and doc.blocks[1].t == "Para" then
      quarto.log.output("Returning first paragraph content")
      return doc.blocks[1].content
    else
      -- Fallback for non-paragraph content
      quarto.log.output("Fallback to stringified content")
      return { pandoc.Str(content) }
    end
  end
end

-- Return the shortcode handler
return {
  ['include_with_indent'] = include_with_indent
}