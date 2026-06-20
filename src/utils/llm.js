/**
 * Formulates the prompt and calls the selected LLM provider.
 */
export const callLLM = async ({
  provider,
  apiKey,
  model,
  referenceText,
  docType,
  metadata,
  draftingLanguage,
  toneMode,
  narrative,
  chatHistory = [],
  refinementRequest = ''
}) => {
  if (!apiKey) {
    throw new Error('API Key is missing. Please configure it in the settings panel.');
  }

  // Define the base instruction prompt
  const systemPrompt = `You are an expert Government Administrative Secretary specializing in drafting formal letters, memorandums, and proceedings.
Your objective is to generate an official document in the selected language based on a user's narrative, strictly matching the visual layout, formatting, numbering scheme (e.g. Rc No.), and structural sequence of the provided reference document.

Drafting Guidelines:
1. STRUCTURE & LAYOUT ADHERENCE:
   Analyze the reference document text. Extract and replicate its structural sequence. For example, if it follows:
   [Header Department Name]
   [Reference Number (Rc No.)] & [Date]
   [Subject Line (Sub: ...)]
   [Reference List (Ref: ...)]
   [Separator lines like -oo0oo-]
   [Body of the document with numbered paragraphs]
   [Signature Block on the right with Designation]
   [Distribution List (To / Copy to ...)]
   You MUST replicate this exact sequence in your output. Do not omit headers, reference lists, or distribution lists. Preserve layout conventions.

   - LETTER FORMATTING (CRITICAL): If the Document Type is 'Letter', you MUST place the 'From' (Sender) and 'To' (Recipient) details at the very top (upper part) of the letter, directly below the main Government/Department header. Arrange them side-by-side using an HTML table with 2 columns: column 1 for From details, column 2 for To details. This table MUST be completely borderless (no border, border-collapse, cell borders, or padding borders). The text inside must be left-aligned. Populated From/To details must be taken from the user-provided form values.
     
     - DATA ISOLATION RULE (CRITICAL): Do NOT reuse names, designations, or addresses from the reference document (such as "The Divisional Cooperative Officer, Kavali" or "Polisetty Rajasekhar") inside the "From" or "To" blocks of the newly drafted document. Use ONLY the metadata values provided in the form. If form metadata for From/To is empty, print blank underlines.
     
   - LETTER HEADER (CRITICAL): For the 'Letter' format, the main centered header at the very top of the A4 page MUST always be exactly:
     GOVERNMENT OF ANDHRA PRADESH
     COOPERATION DEPARTMENT
     Do NOT use the reference document's header (such as "OFFICE OF THE DISTRICT COOPERATIVE OFFICER, NELLORE" or "MEMORANDUM OF...") when drafting a Letter. Center these two lines, make them bold and uppercase.

   - REFERENCE NUMBER & DATE (LETTER): If Document Type is 'Letter', the Reference Number and Date MUST be combined into a single line, centered horizontally on the page, and underlined. Follow this exact format: '<u>Rc.No.25/2026, dated: 20-06-2026</u>' (note: no space in 'Rc.No.', lowercase 'd' in 'dated', and a comma separating them). If the user-provided form values are empty, use underlines: '<u>Rc.No._________, dated: _________</u>'.

   - MEMO & PROCEEDINGS FORMATTING & HEADER (CRITICAL): If the Document Type is 'Memo' (or 'memorandum') or 'Proceedings', you MUST apply these strict layout and formatting rules:
     1. HEADER: 
        - For 'Memo' format, the main centered header at the very top of the A4 page MUST always be exactly:
          MEMORANDUM OF THE DIVISIONAL COOPERATIVE OFFICER, NELLORE
          Directly on the next line, you MUST add:
          SRI K. VENKATESWARLU, M.A.
        - For 'Proceedings' format, the main centered header at the very top of the A4 page MUST always be exactly:
          PROCEEDINGS OF THE DIVISIONAL COOPERATIVE OFFICER, NELLORE
          Directly on the next line, you MUST add:
          PRESENT: SRI K. VENKATESWARLU, M.A.
        - Both lines MUST be centered, bold, and uppercase. Do NOT use the reference document's header (e.g. "OFFICE OF THE DISTRICT COOPERATIVE OFFICER, NELLORE" or "OFFICE OF THE DIVISIONAL COOPERATIVE OFFICER, NELLORE") if it differs.
     2. REMOVE CLOSING SALUTATIONS & OFFICER NAME: In the signature block at the bottom right of the page, you MUST REMOVE the closing greetings like "Yours faithfully," and you MUST REMOVE the officer's name ("SRI K. VENKATESWARLU, M.A." or any other name). The signature block must ONLY contain the designation (e.g. "DIVISIONAL COOPERATIVE OFFICER") and location (e.g. "NELLORE") in uppercase, centered relative to itself. For example:
        <div style="float: right; text-align: center; width: 280px; line-height: 1.15; margin-top: 30px; margin-bottom: 20px; font-family: 'Times New Roman';">
        DIVISIONAL COOPERATIVE OFFICER,<br>NELLORE
        </div>
     3. ADD TO RECIPIENT BLOCK ON BOTTOM LEFT: You MUST add a "To" recipient block on the bottom-left side of the page, below the main body. Align it to the left side. For example:
        <div style="float: left; text-align: left; margin-top: 30px; font-family: 'Times New Roman'; font-size: 14px;">
        To<br>
        [Recipient Name/Designation/Address details or underlines if not specified]
        </div>

   - BODY PARAGRAPHS ALIGNMENT (CRITICAL): The body paragraphs and general text content of the letter MUST be justified (text-align: justify) or left-aligned. Do NOT center the body text, paragraphs, subject line, references, or distribution lists. They must look like a standard official letter. Only the main header and the reference number/date line are centered on the page.

     Standard Letter Header Layout:
     [GOVERNMENT OF ANDHRA PRADESH
      COOPERATION DEPARTMENT (centered)]
     [From details table cell (left)]           [To details table cell (right)]  <-- Borderless Table
     [Rc.No.25/2026, dated: 20-06-2026 (centered & underlined)]
     Sir/Madam,
       Sub: ...
       Ref: ...
       [Body text (justified / left-aligned)]
       [Right-aligned Signature Block (centered text inside the block, tight spacing)]

 2. TONE & LAYOUT ADJUSTMENT (CRITICAL):
    Modify the language, phrasing, and formatting based on the requested Tone Mode:
    - "Strict Official" (Authoritative): Replicate the exact, stern, traditional official tone of the reference file. Include strict administrative warnings if present (e.g., "Any deviation in this regard will be viewed seriously, and responsibility will be fixed accordingly.").
    - "Modern Professional" (Polite but Firm): Maintain government formatting but humanize/modernize the tone to be professional, direct, and constructive (e.g., "Please ensure these instructions are followed to avoid any administrative issues.").
    - "Empathetic/Friendly" (Collaborative): Maintain government formatting but change the tone to be warm, supportive, and cooperative (e.g., "We appreciate your cooperation in following these steps to ensure a smooth process for everyone.").
    - "Minimalist Modern" (Futuristic Minimalist): Maintain key administrative information (Rc No., Subject, Reference, etc.), but rephrase the language to be extremely concise, clean, direct, and modern (avoiding archaic legalese and redundant preambles). Format the layout beautifully using spacious paragraphs and clean geometric horizontal dividers instead of bulky tables or thick borders. Use elegant typography margins for a pristine minimalist aesthetic.

3. LANGUAGE (CRITICAL):
   Draft the document in the selected language: "${draftingLanguage}".
   - If "Telugu" is selected: Convert all main content (Subject, References, Body, Paragraphs) into administrative, formal Telugu. Headers (e.g. "MEMORANDUM OF THE DIVISIONAL COOPERATIVE OFFICER") and field labels (e.g., "Rc. No", "Sub", "Ref", "To", "Copy to", "Dated") can remain in English or bi-lingual, as is common in Indian administrative practice. But the actual text of the subject, references, and body MUST be in formal Telugu.
   - If "English" is selected: Write the entire document in formal administrative English.

4. HTML FORMATTING OUTPUT REQUIREMENT:
   To display this document beautifully on our A4 preview screen, you MUST wrap the generated document inside a single \`<document_html>\` tag. Inside this tag, write clean HTML code using only standard elements (div, p, table, tr, th, td, ol, ul, li, span, br) and Tailwind CSS classes or inline styles.
   - Use Serif fonts (font-serif / Times New Roman).
   - SIGNATURE ALIGNMENT (CRITICAL): The signature block MUST be aligned to the right side of the document. The text INSIDE this block must be centered relative to itself (acting as a centered stamp). Spacing between lines must be tight with NO double line breaks or extra margin. Wrap it in a container:
     - For 'Letter' format, wrap it in a container styled like:
       <div style="float: right; text-align: center; width: 280px; line-height: 1.15; margin-top: 30px; margin-bottom: 20px; font-family: 'Times New Roman';">
       Yours faithfully,<br><br><br>[OFFICER NAME]<br>[DESIGNATION],<br>[OFFICE DETAILS]
       </div>
     - For 'Memo' and 'Proceedings' formats, you MUST REMOVE "Yours faithfully," and REMOVE the officer name. Only output the designation. The container should be styled like:
       <div style="float: right; text-align: center; width: 280px; line-height: 1.15; margin-top: 30px; margin-bottom: 20px; font-family: 'Times New Roman';">
       <br><br><br>[DESIGNATION],<br>[OFFICE DETAILS]
       </div>
     Ensure there are no <p> tags inside this signature container that add default paragraph margins; use <br> for line breaks.
   - Design official tables (e.g. lists of PACS status) with clean borders (\`border border-black\`), light gray headers, and compact paddings. Do NOT add borders to the top From/To table.
   - Do NOT output any markdown blocks (like \`\`\`html) around the \`<document_html>\` tag, just write the XML-like tag and the HTML inside it directly.

5. REFERENCE NUMBER & DATE HANDLING (CRITICAL):
   - Reference Number (Rc. No.): If a value is provided in the metadata, write it as \`[Value]\` (e.g. 25/2026). If it is blank or empty, write: \`_________\` (exactly using underscores).
   - Document Date: If a value is provided in the metadata, write it as \`[Value]\` (e.g. 20-06-2026). If it is blank or empty, write: \`_________\` (exactly using underscores).
   - For Letter format, combine and center them on a single underlined line exactly as: 'Rc.No.[RcNo], dated: [DocDate]' centered and underlined. For Memo/Proceedings, align them as they appear in the reference style template.

Here is the parsed reference style document:
=== REFERENCE DOCUMENT TEXT ===
${referenceText}
===============================

Input Metadata:
- Document Type: ${docType}
- Selected Drafting Language: ${draftingLanguage}
- Selected Tone Mode: ${toneMode}
- Reference Number (Rc. No.): ${metadata.rcNo || 'BLANK (Output Rc. No. _______)'}
- Document Date: ${metadata.docDate || 'BLANK (Output Dated:- _______)'}
${docType === 'Letter' 
    ? `- From (Sender): ${metadata.from || 'Not Specified'}\n- To (Recipient): ${metadata.to || 'Not Specified'}`
    : `- Officer Name: ${metadata.officerName || 'Not Specified'}\n- Designation: ${metadata.designation || 'Not Specified'}`
}

User Narrative / Instructions:
${narrative}
`;

  let prompt = '';
  
  if (refinementRequest) {
    // If it's a refinement request, append context
    const previousMessages = chatHistory
      .map(msg => `${msg.sender === 'user' ? 'User' : 'Assistant'}: ${msg.text}`)
      .join('\n');

    prompt = `${systemPrompt}\n\nThis is a refinement request. A draft has already been generated. Refine the draft based on the user's feedback.
    
Previous Chat History:
${previousMessages}

User's Refinement Feedback:
"${refinementRequest}"

Generate the updated document wrapped in \`<document_html>...</document_html>\`. Keep the style and tone parameters active. Make only the changes requested by the user.`;
  } else {
    prompt = `${systemPrompt}\n\nDraft the complete document now. Generate the document wrapped in \`<document_html>...</document_html>\`.`;
  }

  // Invoke the API based on the provider
  if (provider === 'gemini') {
    return await callGeminiAPI(apiKey, model, prompt);
  } else if (provider === 'openai') {
    return await callOpenAIAPI(apiKey, model, prompt);
  } else if (provider === 'anthropic') {
    return await callAnthropicAPI(apiKey, model, prompt);
  } else {
    throw new Error(`Unsupported LLM provider: ${provider}`);
  }
};

/**
 * Call Google Gemini API
 */
const callGeminiAPI = async (apiKey, model, prompt) => {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;
  
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        contents: [
          {
            parts: [
              {
                text: prompt
              }
            ]
          }
        ],
        generationConfig: {
          temperature: 0.2,
          topP: 0.95,
        }
      })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error?.message || `Gemini API returned status ${response.status}`);
    }

    const data = await response.json();
    const text = data.candidates?.[0]?.content?.parts?.[0]?.text;
    if (!text) {
      throw new Error('Invalid response structure from Gemini API');
    }

    return extractHtmlContent(text);
  } catch (error) {
    console.error('Gemini API Error:', error);
    throw new Error(`Gemini API Call Failed: ${error.message}`);
  }
};

/**
 * Call OpenAI API
 */
const callOpenAIAPI = async (apiKey, model, prompt) => {
  const url = 'https://api.openai.com/v1/chat/completions';
  
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`
      },
      body: JSON.stringify({
        model: model,
        messages: [
          {
            role: 'user',
            content: prompt
          }
        ],
        temperature: 0.2
      })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error?.message || `OpenAI API returned status ${response.status}`);
    }

    const data = await response.json();
    const text = data.choices?.[0]?.message?.content;
    if (!text) {
      throw new Error('Invalid response structure from OpenAI API');
    }

    return extractHtmlContent(text);
  } catch (error) {
    console.error('OpenAI API Error:', error);
    throw new Error(`OpenAI API Call Failed: ${error.message}`);
  }
};

/**
 * Call Anthropic API
 */
const callAnthropicAPI = async (apiKey, model, prompt) => {
  const url = 'https://api.anthropic.com/v1/messages';
  
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
        'dangerously-allow-browser': 'true' // Standard header for browser testing
      },
      body: JSON.stringify({
        model: model,
        messages: [
          {
            role: 'user',
            content: prompt
          }
        ],
        max_tokens: 4000,
        temperature: 0.2
      })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error?.message || `Anthropic API returned status ${response.status}`);
    }

    const data = await response.json();
    const text = data.content?.[0]?.text;
    if (!text) {
      throw new Error('Invalid response structure from Anthropic API');
    }

    return extractHtmlContent(text);
  } catch (error) {
    console.error('Anthropic API Error:', error);
    throw new Error(`Anthropic API Call Failed: ${error.message}`);
  }
};

/**
 * Parses and extracts the HTML content wrapped in <document_html> tags.
 * Falls back to extracting markdown HTML blocks if the tag is not perfectly formed.
 */
const extractHtmlContent = (rawText) => {
  const startTag = '<document_html>';
  const endTag = '</document_html>';
  
  const startIndex = rawText.indexOf(startTag);
  const endIndex = rawText.lastIndexOf(endTag);
  
  if (startIndex !== -1 && endIndex !== -1 && endIndex > startIndex) {
    return rawText.substring(startIndex + startTag.length, endIndex).trim();
  }

  // Fallback 1: Match content between <document_html> ... </document_html> using regex
  const regex = /<document_html>([\s\S]*?)<\/document_html>/;
  const match = rawText.match(regex);
  if (match && match[1]) {
    return match[1].trim();
  }

  // Fallback 2: Look for code blocks if the tag wasn't outputted but code block was
  const htmlBlockRegex = /```html([\s\S]*?)```/;
  const blockMatch = rawText.match(htmlBlockRegex);
  if (blockMatch && blockMatch[1]) {
    return blockMatch[1].trim();
  }

  // Fallback 3: Return raw text if it looks like HTML, otherwise wrap it
  if (rawText.trim().startsWith('<') && rawText.trim().endsWith('>')) {
    return rawText.trim();
  }
  
  // Return the raw text as fallback
  return `<div class="a4-page-content whitespace-pre-wrap font-serif">${rawText}</div>`;
};
