```markdown
# Design System Specification: Chrono-Cyberpunk Editorial
 
## 1. Overview & Creative North Star
**Creative North Star: "The Quantum Ledger"**
 
This design system is engineered for the high-stakes, high-velocity world of binary prediction markets. We are moving away from the "trading app template" and toward a high-fidelity, data-dense editorial experience. The aesthetic—**Chrono-Cyberpunk**—marries the brutalist precision of 1980s mainframe terminals with the ultra-modern fluidity of high-end digital interfaces. 
 
To break the "standard" look, we utilize **Intentional Asymmetry** and **Tonal Depth**. Layouts should prioritize information density without sacrificing visual hierarchy, using monolithic blocks of color and stark, monospace data points to evoke a sense of absolute accuracy and institutional authority.
 
---
 
## 2. Colors: The Obsidian Matrix
Our palette is rooted in an "Obsidian" depth, utilizing high-contrast neon accents to signal binary states (Yes/No) with zero ambiguity.
 
### Color Tokens
*   **Background:** `#0e0e0f` (The Void)
*   **Primary (YES/Cyan):** `#8ff5ff` | **Container:** `#00eefc`
*   **Secondary (NO/Magenta):** `#ff6b9b` | **Container:** `#ba005b`
*   **Tertiary (Profit/Neutral):** `#bcff5f` | **Container:** `#a2f31f`
*   **Surface Hierarchy:**
    *   `surface-container-lowest`: `#000000` (Recessed fields)
    *   `surface-container-low`: `#131314` (Default sections)
    *   `surface-container-high`: `#201f21` (Hover states)
    *   `surface-container-highest`: `#262627` (Active/Elevated layers)
 
### Rules for Application
*   **The "No-Line" Rule:** Prohibit the use of 1px solid borders for sectioning. Boundaries must be defined solely through background color shifts. For example, a trading module (`surface-container-high`) sits directly on the `background` without a stroke.
*   **Surface Nesting:** Treat the UI as a series of stacked obsidian sheets. Use `surface-container-lowest` for input fields to make them feel "carved" into the interface, and `surface-container-highest` for floating trade execution panels.
*   **The "Glass & Gradient" Rule:** Use subtle linear gradients on CTAs (e.g., `primary` to `primary-container`) at a 135° angle. This adds a "synthetic glow" characteristic of cyberpunk aesthetics.
*   **Signature Textures:** Apply a 2% opacity noise texture over large background areas to simulate high-end hardware displays and prevent color banding on dark OLED screens.
 
---
 
## 3. Typography: High-Precision Legibility
We use a dual-typeface system: **Space Grotesk** for structural authority and **Inter** for transactional clarity. 
 
*   **Display & Headlines (Space Grotesk):** These are your "billboard" moments. Use `display-lg` (3.5rem) for market outcomes. The wide tracking and geometric terminals of Space Grotesk should feel like a digital ticker tape.
*   **Data & Labels (Space Grotesk - Monospace feel):** Use for prices, percentages, and timestamps. Monospaced-leaning typography ensures that when numbers change rapidly, the layout does not "jump."
*   **Body & Titles (Inter):** Used for market descriptions and legal fine print. Inter provides the necessary readability for dense paragraphs of market rules.
 
**Identity Logic:** The contrast between the eccentric, tech-leaning Space Grotesk and the invisible efficiency of Inter signals a platform that is both cutting-edge and professionally reliable.
 
---
 
## 4. Elevation & Depth: Tonal Layering
Traditional shadows have no place here. Precision is communicated through light and layering, not "fuzziness."
 
*   **The Layering Principle:** Achieve depth by "stacking." A `surface-container-low` parent should house `surface-container-highest` children for active elements.
*   **Ambient Shadows:** If an element must float (e.g., a modal), use an ultra-diffused shadow: `box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5)`. The shadow should feel like a heavy occlusion rather than a "drop shadow."
*   **The "Ghost Border":** For accessibility in high-density tables, use the `outline-variant` token at **15% opacity**. This creates a "hairline" suggestion of a container without breaking the obsidian-sheet aesthetic.
*   **Glassmorphism:** Use `backdrop-blur: 12px` on top-level navigation bars with a semi-transparent `surface` color. This allows the neon accents of the content to bleed through as the user scrolls, creating a sense of environmental lighting.
 
---
 
## 5. Components: Sharp-Edge Primitives
 
### Buttons
*   **Radii:** Strictly `0px`. Sharp corners imply mathematical precision.
*   **Primary (YES):** `background: primary; color: on-primary;` No border.
*   **Secondary (NO):** `background: secondary; color: on-secondary;` No border.
*   **Tertiary:** Ghost style. No background, `outline-variant` ghost border, `primary` text.
 
### Inputs & Fields
*   **State Logic:** Default state uses `surface-container-lowest`. On focus, the field should trigger a 1px `primary` (Cyan) bottom-border ONLY, simulating a terminal cursor.
 
### Chips (Market Tags)
*   Used for categories (e.g., "Politics", "Crypto"). Use `surface-container-highest` with `label-md` typography. No rounding.
 
### Cards & Lists
*   **Card Separation:** Strictly forbidden to use divider lines. Use a 24px vertical gap or a toggle between `surface-container-low` and `surface-container-highest` to differentiate items.
*   **Data Density:** Lists should favor a "Spreadsheet-Plus" look. Right-align all monospaced numerical data for easy vertical scanning.
 
### Additional Component: The "Binary Toggle"
*   A custom component for prediction markets. A split-block button where the "YES" half is `primary` and the "NO" half is `secondary`. The active side grows slightly in width (Asymmetry), indicating the current market weight or user selection.
 
---
 
## 6. Do's and Don'ts
 
### Do:
*   **Do** use extreme vertical rhythm. Use large gaps (64px+) between major sections but tight, dense spacing (4px-8px) within data clusters.
*   **Do** use `primary` (Cyan) for success/positive trends and `secondary` (Magenta) for failure/negative trends.
*   **Do** right-align all monospaced numbers in tables to ensure decimal points align.
 
### Don't:
*   **Don't** use border-radii. Even a 4px radius softens the "Chrono-Cyberpunk" edge we are aiming for.
*   **Don't** use pure white (`#FFFFFF`) for long-form body text; use `on-surface-variant` (`#adaaab`) to reduce eye strain on the obsidian background.
*   **Don't** use standard "Success Green." In this system, `tertiary` (Lime) is for neutral/profit, while `primary` (Cyan) is the core action color.
 
---
 
**Director’s Final Note:** This system is about the tension between the dark void of the background and the electric urgency of the data. Keep it sharp. Keep it fast. If it looks like a standard dashboard, you've added too many borders. Let the colors and the type breathe.```