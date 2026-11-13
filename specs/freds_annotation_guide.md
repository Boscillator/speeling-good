# Annotation Guide

## Span editing
- First identify if the model has correctly flagged the location where the error occurse.
- We want the mimimal span that includes the whole word with the error.
- The most common error is for the model to include to much context.
  * For example, the model may suggest "a tees worth" -> "a tee's worth",
    however, "tees" -> "tee's" is the more mimial example, so the span should
    be corrected.
- The second most common error is to include punctuation in the span.
  * "didnt." -> "didn't." should be marked as "didnt" -> "didn't".
- Merged and duplicated words should include both words in the span.
  * "asbatteries" -> "as batteries" and "the the" -> "the" are both correct spans.
- Even if the suggestion is out of scope or not an error, still correct the span
  as if it was.
- When in doubt, think about what a spell checker would highlight and where the
  token bounderies are.

If the span is incorrect, you will be asked to annotate the corrected span. 
Select "corrected_span", then highlight the approprite span. Do undo your work
select it in the bottom right panel and then delete in the top right.

## Inclusion
- Select "not_an_error" when the initial word was correct in context.
  * Puns, unusual words, etc are not an error.
  * Casual language is not an error (e.g. "til", "48hrs")
  * Expressive language is not an error (e.g. "I ate toooo much!" should not be edited to use "too")
- Select "out_of_scope" for things that are errors, but should not be flagged.
  * The most common example is corrections that are two long. If a correction
    takes more than two-three words to flag, that is moving closer to the role
    of a grammer checker and is out of scope.
  * If you can determine from context what a proper noun was supposed
    to be, even with the help of google, that should be marked as a
    "in_scope". However, if it is simply an uncommon spelling of a name,
    or you are unable to decide on base on context, it should be marked as
    "out_of_scope".
- Select "in_scope" for edits we want to correct.
  * Obvious spelling mistakes, including those where the incorrect word is a
    real dictionary word, but wrong from context and an easy typo away.
  * Inflection problems (e.g. tense "lead" -> "led", plural agreement
    "member" -> "members", general inflection "hair legs" -> "hairy legs")
  * Adjecent word repetitions (e.g. "the the" -> "the").
  * Whitespace errors (e.g. "asbatteries" -> "as batteries", "Video.I" -> "Video. I")
  * Clear typos of common names (e.g. "Rupurt Murodch" -> "Rupurt Murdoch",
    "tumblr" -> "Tumblr")
- Select "unsure" when you cannot determind from context if the work is correct
  or there apperse to be a techical error.

## Replacement
- Did the model get the correct answer? Only shown if the error was "in_scope".
- If it did not, please correct it respecting the corrected span.
- If there were multiple error (e.g. spelling and plural agreeement in the same
  word) fix both of them in the replacement.

## Predicted correction type
- This is not super important to get write, but it is good to provide feedback.
- The most common errors are mixing up style and grammar.
  * Style should be for cases where it could be right. Typically these will be
    marked out of scope.

