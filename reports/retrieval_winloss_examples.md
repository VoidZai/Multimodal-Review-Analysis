# Retrieval win/loss examples (T3.8; PLAN.md §7 appendix)

Questions where BM25 and dense retrieval disagreed sharply on Recall@5 (CRAGB v1). ✅ marks a hit that is actually in the question's pooled `relevant_ids`; ❌ marks a hit that is not.

## BM25 wins, dense loses

### `colour_appearance_006` (colour_appearance)
**Q:** Are there any reviews that discuss how the product's color changes in different lighting conditions?

Recall@5: BM25=0.67, dense=0.00 — gold relevant ids: 47004, 61018, 75274, 102131, 107528, 171235

**BM25 top-5:**
- ✅ `171235` (score 26.944): Beautiful, delicate necklace and chain. Light, airy. The stones (says they're real) sparkle nicely in different lighting conditions. The chain is the right length to show off this gorgeous trinity kno
- ✅ `61018` (score 23.077): I ordered these earrings because I was attracted to the mosaic colors. They came nicely packaged with a soft cloth. They appear substantial and well made. Unfortunately, I was disappointed in the lack
- ❌ `124005` (score 22.669): This is a great headband. The color I received looks like pink lemonade. The fabric is very soft and won't damage hair in any way. It's ideal for curly hair. There are instructions included that tell 
- ✅ `75274` (score 21.522): This tie changes character in different lighting in a very nice way, from midnight to baby blue with just the right sheen. The textures are detailed and evident but not jumping off the tie. Material s
- ✅ `47004` (score 21.454): I have the “Ice Blue Beads” color of the Alex and Ani Splendid Beaded Expandable Bangle. I like this color because it looks different depending on how the light hits it. Under natural lighting, the be

**Dense top-5:**
- ❌ `186910` (score 0.756): I bought 2. A darker/brighter on and a softer colored one. It is a lighter shade in pastels to the brighter colored one. I thought they were patterned differently. The pictures showed them folded or d
- ❌ `167340` (score 0.750): Product is excellent, but the colors come alittle bit different from advertised!
- ❌ `169256` (score 0.747): Definitely, color is an issue should have waited and purchased the lighter neutral color. Will wait for colder weather for a more comprehensive review.
- ❌ `22288` (score 0.745): It's aight. Not what I thought it was gone be. Ion be out at night so want get to enjoy color changing.
- ❌ `197015` (score 0.735): I do like the product overall but I am extremely disappointed with the actual color. I purchased the pink and turquoise set. The blue isn’t vivid at all (as depicted in the pictures). Instead it’s a d

**Why:** BM25 matched the literal phrase “different lighting conditions” appearing almost verbatim in 4 of its top-5 reviews; dense's nearest neighbors were generic color-disappointment reviews that never mention lighting at all, so it missed the lighting-specific angle entirely.

### `colour_appearance_neg_000` (colour_appearance)
**Q:** Does the dye lot number affect the shade of colour received?

Recall@5: BM25=0.50, dense=0.00 — gold relevant ids: 28047, 134904

**BM25 top-5:**
- ❌ `83691` (score 26.773): It does fit but the colour is completely different from the picture here it is a lot more fuschia and almost neon colour like orange even not flattering at all. Not wearing this probably ever, maybe j
- ❌ `90174` (score 26.652): These are very nice and long. I love the vivid colour. The only thing I didn't care for was the strong odor, which I'm sure was the dye. The smell did wash out.
- ❌ `105003` (score 25.668): These area so very comfortable and fits very nicely. No toe pinch or rub. TBN the colour in my picture is different than the stock picture, however the colour of the shoe is exactly as pictured in the
- ❌ `3036` (score 25.653): Received the wrong colour twice from the seller, so I returned them both. Had I not been keen on the colour sent to me, I would have kept them. They are incredibly squishy and soft.
- ✅ `28047` (score 25.045): One shoe changes colour more than the other does.

**Dense top-5:**
- ❌ `191558` (score 0.733): Buying a bunch if colors.
- ❌ `7610` (score 0.731): rings changed colors
- ❌ `63699` (score 0.730): I purchased 3 packages of 8 panties. I was sent packages containing three panties. While the total number sent is correct, this is not what I ordered. In a package of 8 there will be a greater variety
- ❌ `167340` (score 0.728): Product is excellent, but the colors come alittle bit different from advertised!
- ❌ `119078` (score 0.728): Does not file as expected but nice color

**Why:** BM25's heavy weighting on “colour”/“dye” surfaced one genuinely relevant review (a shoe's dye lot varying between the two feet) purely on term overlap; dense clustered around generic “wrong color than expected” complaints and missed the dye-lot-specific review.

### `fit_sizing_neg_000` (fit_sizing)
**Q:** Do buyers report the exact chest measurement in inches for the size medium?

Recall@5: BM25=0.50, dense=0.00 — gold relevant ids: 47706, 110475, 116514, 125710

**BM25 top-5:**
- ❌ `41628` (score 30.129): The fit is really off. The body is slightly fitted. The chest measurement on a large is 21 inches. The sleeve opening is too wide. It measures 7 1/2 inches on the large. You need Popeye arms for this 
- ❌ `17016` (score 27.895): This lounge undershirt from ATTI TUDE is very soft. It makes a great base layer or a comfortable shirt to wear around the house and to bed. The modal fabric sort of hugs your body, so the shirt moves 
- ✅ `116514` (score 26.956): This base layer set comes in a nylon zipper pouch that you could reuse to keep the set in a backpack or to pack it down for traveling. When I took the set out of the pouch, I was worried the shirt and
- ❌ `37161` (score 26.734): The sizing information in the product listing is completely wrong. I normally wear a 44L jacket. The second picture in the product listing indicates that for size XL, chest is 50.8 inches, length is 2
- ✅ `47706` (score 26.694): First of all, I love the lingerie set. The lace material isn't a bother and it compliments the body quite well. I did order the medium size and the fit was a bit on the loose side. Most likely going t

**Dense top-5:**
- ❌ `12416` (score 0.767): Bought a medium within the specified chest dimension , what I got is way too small, you would have to be severely anorexic to fit in it
- ❌ `52525` (score 0.757): I order an x-large and I believe it was miss labeled and it was a medium. The refund process was great.
- ❌ `33401` (score 0.747): Falsifying the size it didn’t measure what it says on the website
- ❌ `176528` (score 0.738): I received the correct size however it was WAY smaller than advertised
- ❌ `68453` (score 0.737): I'm a 34 and it says I should purchase Mediums, so I did. They're too small.

**Why:** BM25's exact match on “chest”/“measurement”/“inches” found reviews that literally cite numeric measurements; dense's notion of “size complaint” pulled generic “ran small” reviews with no numbers in them at all, missing what the question actually asked for.

### `durability_neg_000` (durability)
**Q:** How many wash cycles on average does the product survive under controlled lab testing?

Recall@5: BM25=0.50, dense=0.00 — gold relevant ids: 78415, 93761, 121719, 187723

**BM25 top-5:**
- ❌ `191398` (score 24.634): This product did not survive a single washing. Threads through the waist unraveled and garroted other clothes in the wash.
- ❌ `147473` (score 21.362): These socks are super comfortable! They are perfect for wearing around the house during the winter. The creatures are so cute and adorable. The socks also fit perfectly and can survive many washes. I 
- ✅ `78415` (score 20.310): Authentic Pigment makes the absolute best t shirts. No fade whatsoever for about 50 wash cycles.
- ❌ `150901` (score 19.645): The fit is good - not too baggy or restrictive. the support is average. very soft to touch, no riding up or bunching. seems to more breathable than Reebok, Columbia and Champion (though I like the Cha
- ✅ `93761` (score 18.633): They fit well and arrived quickly, I wash in cold water and the blue one still looks good, but the black is noticably faded after less than 10 cycles.

**Dense top-5:**
- ❌ `3896` (score 0.721): We will see how long they last after washing them for 3 to 6 months
- ❌ `55127` (score 0.719): Holding up well after multiple washings.
- ❌ `27799` (score 0.719): Holding up well after multiple washings.
- ❌ `119035` (score 0.716): Hold up well in the wash
- ❌ `65553` (score 0.713): So far it’s washed well and held up

**Why:** BM25 found reviews using the near-literal phrase “wash cycles” with an attached count (“50 wash cycles”, “less than 10 cycles”); dense retrieved topically-similar “holds up after washing” reviews, but none of its top-5 actually state a cycle count.

## Dense wins, BM25 loses

### `defects_neg_001` (defects)
**Q:** Do buyers report the specific factory batch number associated with defective units?

Recall@5: BM25=0.00, dense=1.00 — gold relevant ids: 117931

**BM25 top-5:**
- ❌ `182748` (score 19.413): Defective, it has a large 8&#34; faded ring all the way around the top of the Beret. Why do we keep receiving defective merchandise !
- ❌ `110899` (score 18.700): The sheepskin inside one of the slippers is defective. It is clumped, with a ball of clump, and misshapen. This is now the second defective pair I received. I will not trying a third. I do not recomme
- ❌ `17998` (score 17.736): Socks with non slip properties, but not the best, but neither the worst. Sorry for the inconclusive report, it is just that they do not excelled, neither fail in their anti skid duties.
- ❌ `117466` (score 17.678): Really disappointed with the appearance of the Firelily in black leather. As another reviewer mentioned, the front of the shoes looks scuffed. The overall pattern seems flawed in the front of both sho
- ❌ `175995` (score 17.484): I like the soft feel of the bangle and tassels though it is large and impractical for a key ring unless it was for a specific purpose, like laundry room keys, etc. The fringed tassel is perfect though

**Dense top-5:**
- ❌ `20772` (score 0.708): Wrong measurement from Factory
- ❌ `131204` (score 0.708): I’ve purchased many 505s in the past but these seem to have been mislabeled
- ❌ `193820` (score 0.701): Just cannot seem to trust whether these are counterfeit, defective, factory 2nds, or actual Levi's anymore. My Levi's when ordered online too often have flaws: improper fit, bad zipper, 5 belt loops i
- ✅ `117931` (score 0.690): I ordered these twice. The first one was give N go as advertised. When the second batch arrived it was labeled give N go 2.0. The 2.0 according to other reviews has not had a good rating. I like the o
- ❌ `147902` (score 0.689): Ordered small and shipped large. Although Amazon shipped the correct produce, I retain the negative review as I have recently received 2 wrong products and 1 that didn't work. This is not the Amazon I

**Why:** Dense's one relevant hit is only a loose semantic reach (a review mentioning a product's appended version number, not literally a factory batch number) — still enough to edge out BM25, whose top-5 all keyword-matched “defective” broadly without touching batch/version numbering at all.

### `occasion_neg_000` (occasion)
**Q:** Is this item officially licensed or endorsed for use at professional sporting events?

Recall@5: BM25=0.00, dense=0.50 — gold relevant ids: 83107, 173986

**BM25 top-5:**
- ❌ `120269` (score 23.045): Love this top! I typically wear it to sporting events or just running around doing errands. It's comfy & fits well.
- ❌ `26122` (score 20.205): Good quality for the price. Works well for stadium, concert and sporting events. Just the right size for small wallet, cell phone & sunglasses.
- ❌ `44057` (score 20.076): I love to use tote bags and this one is really cute! I like the see through quality of it. While the product description says that this can be used for sporting events, it's too large for most of the 
- ❌ `76910` (score 16.990): What's not to love?<br />It's a belt with a bear on the buckle, holds your pants or trousers up (sparing you embarrassment of possibly being bear back, bad pun, but I had to), the price is reasonable 
- ❌ `19384` (score 16.721): I was very pleased with this product. The material is thick, high quality, and felt like a professional chef. It was comfortable to wear all day while I made Thanksgiving dinner. Made for a great memo

**Dense top-5:**
- ❌ `106505` (score 0.707): Its good for going to sporting events. They won't let any bag that isn't clear. I can fit a jacket and some smaller stuff in there. It's pretty cheap though. The rope already came untied after a few u
- ❌ `135939` (score 0.685): Since public venues are now so security conscious, I was looking for a clear tote bag to use should we ever go to a baseball game or large event where clear bags are required. This is made of a nice h
- ✅ `173986` (score 0.682): Counterfeit item not really FOX Racing
- ❌ `163660` (score 0.681): This product was perfect for my athletes. Baseball and football players and they loved it. Great value for the money.
- ❌ `177203` (score 0.680): I use this bag for stadium use. It is clear so they can see what is inside.

**Why:** Dense surfaced a review calling the item “counterfeit, not really [brand]” — a real semantic link to “not officially licensed” with zero shared vocabulary with the question; BM25 matched the literal phrase “sporting events” but only found reviews about using the product at events, not about licensing or endorsement.

### `defects_005` (defects)
**Q:** Are there any complaints about defects or damage to the packaging?

Recall@5: BM25=0.10, dense=0.50 — gold relevant ids: 3574, 42696, 57010, 65646, 74543, 125294, 140332, 150678, 164069, 171382

**BM25 top-5:**
- ❌ `114050` (score 19.541): After replacing the first pair (the right boot squeaked relentlessly) I now have a functional pair of these boots. There was no hassle or complaints about the exchange. The boots I currently have are 
- ❌ `90760` (score 19.309): Super cute and comfy! They seem true to size (it's hard to find a 10.5 so I'm happy about that)<br />The only complaint I have is the mildew on the boots, I mean... WTH?? It cleaned up just fine thoug
- ❌ `193494` (score 19.234): sturdy and pretty but some black keys are scratched or faded. the lid covers the defects, though.
- ❌ `66445` (score 18.584): Favorite brand of shoe. I can generally always count on their fit. I have and have had a least a dozen pairs of different styles over the years. Ordered the 8.5 2E. They fit perfectly. Comfy. I have w
- ✅ `57010` (score 18.296): Product works as intended, but nothing about the product or packaging was too interesting

**Dense top-5:**
- ✅ `171382` (score 0.766): Produced arrived cracked even though box seemed fine.
- ✅ `125294` (score 0.761): The product itself is not bad. It keeps my ear warm even when the weather is below 0 Fahrenheit. However, the packaging needs extra work to be good. It looked like used when it arrived and that shaken
- ✅ `3574` (score 0.759): Item pretty decent, as expected but packaging....not sure. Is this how it’s supposed to be or has this been repacked? Used and returned item???
- ✅ `140332` (score 0.759): We open the package with a glue on it. Damaged before use
- ✅ `150678` (score 0.755): OK, a little cheaply made and box had some damage, but item was not damaged

**Why:** Dense's embedding captured “packaging damage” as a coherent semantic cluster — 4 of its top-5 hits are directly about damaged/suspicious packaging; BM25's top hits matched “defect”/“complaint” generically (boots, general quality gripes) and mostly missed the packaging-specific angle, since “packaging” itself wasn't a dominant term in most of the actually-relevant reviews.

### `colour_appearance_010` (colour_appearance)
**Q:** Do reviewers mention any issues with the color being uneven or inconsistent?

Recall@5: BM25=0.00, dense=0.33 — gold relevant ids: 59714, 69704, 195703

**BM25 top-5:**
- ❌ `105388` (score 22.779): I agree with other reviewers that the sizing on these shoes is inconsistent. I ordered 2 pairs, 1 leather and the other canvas. The leather pair fits fine and I am wearing them. The canvas pair in the
- ❌ `85990` (score 22.081): The straps of these shoes are way too wide you practically slip off them if you do any walking on uneven ground or cobbelstones.
- ❌ `11674` (score 21.113): I am a pretty standard medium and these felt more like a small especially in the waist. Otherwise they are well made and comfortable. My recommendation is to size up. Unlike other reviewers, I didn't 
- ❌ `186699` (score 20.965): The shirts are really thin and the graphics are very light, bordering on faded-looking. The hems are uneven, being longer at the sides than front or back. And even though it is listed as having free r
- ❌ `25768` (score 20.896): This hoodie is lightweight and works well for hot weather. It feel soft, comfortable, cool on the skin. I like the fit and feel of the sweatshirt.<br /><br />Other reviewers have flagged issues with t

**Dense top-5:**
- ❌ `113041` (score 0.772): Reviewers seem happy with normal colors…I expected the top to have the color I chose and shiny texture as shown, did not, returned. (Pon?)
- ❌ `169256` (score 0.759): Definitely, color is an issue should have waited and purchased the lighter neutral color. Will wait for colder weather for a more comprehensive review.
- ✅ `195703` (score 0.744): Looks good but it’s fading maybe mine is defective I don’t know
- ❌ `68680` (score 0.741): The quality wasn’t what I was hoping for but the color and size was right.
- ❌ `134096` (score 0.740): The colors didn’t match what I wanted it to

**Why:** BM25 matched “uneven”/“inconsistent” literally, but landed on sizing and hem complaints — a lexical false-positive, since those reviews use the same words for a different topic; dense at least found a loosely related color-fading review (“maybe mine is defective”) that shares no vocabulary with the question but is actually about color.
