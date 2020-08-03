# How To Ask

[How-To-Ask-Questions-The-Smart-Way](https://github.com/ryanhanwu/How-To-Ask-Questions-The-Smart-Way/blob/master/README-zh_CN.md)

> when asking questions on the net such as stack overflow or another, you should know how to ask a good question in english

回答者们大多是自愿的，倾向于回答那些真正提出有意义问题，并且愿意主动参与解决问题的人。

## 提问之前

提问之前，尝试在google搜索，尝试在官方文档搜索，尝试阅读源码

在google搜索时，可以采用 `python packagename errormessage` 等类似的方式。

提问时可以加上`I googled on the following phrase but didn't get anything that looked promising`

另外，用`“Would someone provide a pointer?”, “What is my example missing?”, and “What site should I have checked?”` 比 `“Please post the exact procedure I should use.”`好很多，前者表明你只需要有人指明方向，你有完成的能力和信心。

## 提问时

### 选择提问的网站

提问时不要犯以下错误：
* 主题不合
* 重复问题
* 在探讨进阶技术问题的论坛张贴非常初级的问题；反之亦然
* 向既非熟人也没有义务解决你问题的人发送私人电邮

### stack overflow

* Super User 是问一些通用的电脑问题，如果你的问题跟代码或是写程序无关，只是一些网络连线之类的，请到这里。
* Stack Overflow 是问写程序有关的问题。
* Server Fault 是问服务器和网管相关的问题。

### 网站和IRC论坛

通过论坛或者IRC频道来提供使用者支持服务有增长的趋势，IRC问问题时不要长篇大论。

### 使用项目邮件列表

查看项目的邮件列表，或者新闻列表，注册新闻列表等及时收到项目的最新情况，问题通过前面的途径解决不了，可以给开发者邮件列表发送邮件。

### 使用有意义且描述明确的标题

不要在标题中写`Please help me`或者`PLEASE HELP ME!!!!`，这种标题会被忽略，标题中应该是简明扼要的提出问题。

一个好的例子是`object - deviation`式的描述，`object`指出哪一部分东西有问题，`deviation`描述了与期望的行为不一致的地方。

Stupid:
> HELP! Video doesn't work properly on my laptop!

Smart:
> X.org 6.8.1 misshapen mouse cursor, Fooware MV1005 vid. chipset

Smarter:
> X.org 6.8.1 mouse cursor on Fooware MV1005 vid. chipset - is misshapen

尽量不要在其他人的讨论那里重新开始一个新的提问，因为这样只会让特定的人看到你的问题，除非你就是要这样做。

### 使问题易于回复

要求回复到电子邮件时非常不礼貌的，如果你只是想通过邮件知道其他人是否回答，可以在提问的网站设置提醒。

### `Write in clear, grammatical, correctly-spelled language`

粗心的提问者也会粗心的写程序与思考，粗心的问题不太可能得到回答。

正确的拼写、标点符号和大写也是重要的。不要全部用大写，这被视为是不礼貌的嚷嚷。

如果英语不是母语，提示潜在的回答者你有潜在的语言困难也是很好的。比如：

* 英文不是我的母语，请原谅我的错字或语法。
> English is not my native language; please excuse typing errors.
* 我对技术名词很熟悉，但对于俗语或是特别用法比较不甚了解。
> I am familiar with the technical terms, but some slang expressions and idioms are difficult for me.

### 使用易读取且标准的文件格式发送问题

使用纯文字而不是html，上传附件是可以的，但是不要上传封闭格式编写的文件，如word和excel。

论坛上也不要过分使用表情、色彩和字体。

### 精确的描述问题并言之有物

* Describe the symptoms of your problem or bug carefully and clearly.

* Describe the environment in which it occurs (machine, OS, application, whatever). Provide your vendor's distribution and release level (e.g.: “Fedora Core 7”, “Slackware 9.1”, etc.).

* Describe the research you did to try and understand the problem before you asked the question.

* Describe the diagnostic steps you took to try and pin down the problem yourself before you asked the question.

* Describe any possibly relevant recent changes in your computer or software configuration.

* If at all possible, provide a way to reproduce the problem in a controlled environment.

Do the best you can to anticipate the questions a hacker will ask, and answer them in advance in your request for help.

### 话不在多而在精

需要提供尽可能精简的信息，而不是把成堆的出错代码或者资料完全转录到你的提问之中，如果可以重现错误的场景，尽量裁剪的越小越好。

### 别动辄声称找到bug

在使用软件的时候，你发现的问题是否可以重现，是否有其他人也遇到过这个问题，你发现的问题可能是你弄错了，而不是软件本身有问题。

编写软件的人总是非常辛苦地使它尽可能完美。如果你声称找到了 Bug，也就是在质疑他们的能力，即使你是对的，也有可能会冒犯到其中某部分人。当你在标题中嚷嚷着有Bug时，这尤其严重。

### 描述问题症状而非你的猜测

告诉黑客们你认为问题是怎样造成的并没什么帮助。因此要确信你原原本本告诉了他们问题的症状，而不是你的解释和理论；让黑客们来推测和诊断。如果你认为陈述自己的猜测很重要，清楚地说明这只是你的猜测，并描述为什么它们不起作用。

### 按时间先后列出问题症状

问题发生前的一系列操作，往往是对找到问题最有帮助的线索，因此，你的说明里应该包含你的操作步骤，以及机器和软件的反应，直到问题发生。在命令行处理的情况下，提供一段操作记录（例如运行脚本工具所生成的），并引用相关的若干行（如 20 行）记录会非常有帮助。

### 描述目标而不是过程

如果你想弄清楚如何做某事（而不是报告一个 Bug），在开头就描述你的目标，然后才陈述重现你所卡住的特定步骤。

### 别要求使用私人电邮回复

黑客们认为问题的解决过程应该公开、透明，此过程中如果更有经验的人注意到不完整或者不当之处，最初的回复才能够、也应该被纠正。同时，作为提供帮助者可以得到一些奖励，奖励就是他的能力和学识被其他同行看到。

### 去掉无意义的提问句

避免用无意义的话结束提问，例如`“Can anyone help me?” ` ,`“Is there an answer?”`

首先：如果你对问题的描述不是很好，这样问更是画蛇添足。
其次：由于这样问是画蛇添足，黑客们会很厌烦你。而且通常会用逻辑上正确，但毫无意义的回答来表示他们的蔑视， 例如：`“Yes, you can be helped”` `“No, there is no help for you.”`

### 即使你很急也不要在标题写 “Urgent”

有人会直接把这个当做关键词过滤掉。

### Courtesy never hurts, and sometimes helps

Be courteous. Use “Please” and “Thanks for your attention” or “Thanks for your consideration”. Make it clear you appreciate the time people spend helping you for free.

### 问题解决后，加个简短的补充说明

问题解决后，向所有帮助过你的人发个说明，让他们知道问题是怎样解决的，并再一次向他们表示感谢。如果问题在新闻组或者邮件列表中引起了广泛关注，应该在那里贴一个说明比较恰当。

最理想的方式是向最初提问的话题回复此消息，并在标题中包含`FIXED`，`RESOLVED`或其它同等含义的明显标记。

## 如何解读答案

### RTFM and STFW: How To Tell You've Seriously Screwed Up

RTFM （Read The Fucking Manual），基本上他是对的，你应该取读一读。

STFW（Search The Fucking Web），多半也是对的，你应该上网搜索一下，更温和的说法是 Google is your friend!

### If you don't understand...

如果你看不懂回应，别立刻要求对方解释。像你以前试着自己解决问题时那样（利用手册，FAQ，网络，身边的高手），先试着去搞懂他的回应。如果你真的需要对方解释，记得表现出你已经从中学到了点什么。

For example, suppose I tell you: “It sounds like you've got a stuck zentry; you'll need to clear it.” Then: here's a bad followup question: “What's a zentry?” Here's a good followup question: “OK, I read the man page and zentries are only mentioned under the -z and -p switches. Neither of them says anything about clearing zentries. Is it one of these or am I missing something here?”

### Dealing with rudeness

很多黑客圈子中看似无礼的行为并不是存心冒犯。相反，它是直接了当，一针见血式的交流风格，这种风格更注重解决问题，而不是使人感觉舒服而却模模糊糊。

如果你觉得被冒犯了，试着平静地反应。如果有人真的做了出格的事，邮件列表、新闻群组或论坛中的前辈多半会招呼他。如果这没有发生而你却发火了，那么你发火对象的言语可能在黑客社区中看起来是正常的，而你将被视为有错的一方，这将伤害到你获取信息或帮助的机会。

另一方面，你偶尔真的会碰到无礼和无聊的言行。与上述相反，对真正的冒犯者狠狠地打击，用犀利的语言将其驳得体无完肤都是可以接受的。然而，在行事之前一定要非常非常的有根据。

记着：当黑客说你搞砸了，并且（无论多么刺耳）告诉你别再这样做时，他正在为关心你和他的社区而行动。对他而言，不理你并将你从他的生活中滤掉更简单。

## Questions Not To Ask

* Q: Where can I find program or resource X?
* Q: How can I use X to do Y?
* Q: How can I configure my shell prompt?
* Q: Can I convert an AcmeCorp document into a TeX file using the Bass-o-matic file converter?
* Q: My {program, configuration, SQL statement} doesn't work
* Q: I'm having problems with my Windows machine. Can you help?
* Q: My program doesn't work. I think system facility X is broken.
* Q: I'm having problems installing Linux or X. Can you help?
* Q: How can I crack root/steal channel-ops privileges/read someone's e-mail?

## Good and Bad Questions

Stupid: Where can I find out stuff about the Foonly Flurbamatic?
This question just begs for "STFW" as a reply.

Smart: I used Google to try to find “Foonly Flurbamatic 2600” on the Web, but I got no useful hits. Can I get a pointer to programming information on this device?
This one has already STFWed, and sounds like there might be a real problem.

Stupid: I can't get the code from project foo to compile. Why is it broken?
The querent assumes that somebody else screwed up. Arrogant git...

Smart: The code from project foo doesn't compile under Nulix version 6.2. I've read the FAQ, but it doesn't have anything in it about Nulix-related problems. Here's a transcript of my compilation attempt; is it something I did?
The querent has specified the environment, read the FAQ, is showing the error, and is not assuming his problems are someone else's fault. This one might be worth some attention.

Stupid: I'm having problems with my motherboard. Can anybody help?
J. Random Hacker's response to this is likely to be “Right. Do you need burping and diapering, too?” followed by a punch of the delete key.

Smart: I tried X, Y, and Z on the S2464 motherboard. When that didn't work, I tried A, B, and C. Note the curious symptom when I tried C. Obviously the florbish is grommicking, but the results aren't what one might expect. What are the usual causes of grommicking on Athlon MP motherboards? Anybody got ideas for more tests I can run to pin down the problem?

## 更好的回答问题

态度和善一点。问题带来的压力常使人显得无礼或愚蠢，其实并不是这样。

对初犯者私下回复。对那些坦诚犯错之人没有必要当众羞辱，一个真正的新手也许连怎么搜索或在哪找常见问题都不知道。

如果你不确定，一定要说出来！一个听起来权威的错误回复比没有还要糟，别因为听起来像个专家很好玩，就给别人乱指路。要谦虚和诚实，给提问者与同行都树个好榜样。

如果帮不了忙，也别妨碍他。不要在实际步骤上开玩笑，那样也许会毁了使用者的设置 —— 有些可怜的呆瓜会把它当成真的指令。

试探性的反问以引出更多的细节。如果你做得好，提问者可以学到点东西 —— 你也可以。试试将蠢问题转变成好问题，别忘了我们都曾是新手。

尽管对那些懒虫抱怨一声 RTFM 是正当的，能指出文件的位置（即使只是建议个 Google 搜索关键词）会更好。

如果你决定回答，就请给出好的答案。当别人正在用错误的工具或方法时别建议笨拙的权宜之计（wordaround），应推荐更好的工具，重新界定问题。

正面的回答问题！如果这个提问者已经很深入的研究而且也表明已经试过 X 、 Y 、 Z 、 A 、 B 、 C 但没得到结果，回答 试试看 A 或是 B 或者 试试 X 、 Y 、 Z 、 A 、 B 、 C 并附上一个链接一点用都没有。

帮助你的社区从问题中学习。当回复一个好问题时，问问自己如何修改相关文件或常见问题文件以免再次解答同样的问题？，接着再向文件维护者发一份补丁。

如果你是在研究一番后才做出的回答，展现你的技巧而不是直接端出结果。毕竟授人以鱼不如授人以渔。