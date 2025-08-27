<template>
	<view class="container">

		<!-- 顶部导航栏 -->
		<view class="nav-bar">
			<image class="nav-icon" src="../../static/history.png" mode="widthFix" @tap="toggle('left')"></image>
			<text class="nav-title">新对话</text>
			<image class="nav-icon" src="../../static/new.png" mode="widthFix" @tap="newChat"></image>
		</view>

		<!-- 显示对话记录列表 -->
		<view class="message-container" v-if='chatlists.length > 0'>

			<!-- 占位符 -->
			<view class="nav-bar-empty"> </view>

			<template v-for="(chat, ind) in chatlists">

				<!-- 用户消息 -->
				<view class="message-item reply-message" v-if="chat.role == 'user'">
					<view class="message-content">
						<text class="message-text">{{chat.content}}</text>
					</view>
				</view>

				<!-- 回复消息 -->
				<view class="message-item user-message" v-if="chat.role == 'assistant'">
					<image class="avatar" src="/static/response.png" mode="aspectFill"></image>
					<view class="message-content wbg">
						<zero-markdown-view :markdown="chat.content" :aiMode='true'></zero-markdown-view>
						<!-- <text class="message-text">{{chat.content}}</text> -->
					</view>
				</view>
			</template>

			<!-- 占位符 -->
			<view class="tmp-bottom"> </view>

		</view>


		<!-- 中间 Logo 和提示语 -->
		<view class="main-content" v-else>
			<image class="logo" src="../../static/ds.png" mode="widthFix"></image>
			<text class="welcome">你好呀！我是 DeepSeek</text>
			<text class="desc">
				我可以和你聊天、答疑、搜索，放心交给我吧~
			</text>
		</view>

		<!-- 底部输入框 -->
		<view class="input-area">
			<input class="msg-input" v-model="msg" placeholder="给 DeepSeek 发送消息"
				placeholder-class="input-placeholder" />
			<view class="input-options">
				<view class="option" @tap="deepThink" :style="{backgroundColor:dt_bgc}">
					<image class="option-icon" src="../../static/dt.png" mode="widthFix"></image>
					<text class="option-text">深度思考 (R1)</text>
				</view>
				<view class="option">
					<image class="option-icon" src="../../static/web.png" mode="widthFix"></image>
					<text class="option-text">联网搜索</text>
				</view>
				<view class="file">
					<image class="add-icon" src="../../static/add.png" mode="widthFix"></image>
				</view>
				<view class="send" @tap='sendMsg'>
					<image class="send-icon" src="../../static/send.png" mode="widthFix"></image>
				</view>
			</view>
		</view>

		<!-- 弹出窗口 -->
		<view>
			<!-- 普通弹窗 -->
			<uni-popup ref="popup" background-color="#fff" @change="change">
				<view class="popup-content" :class="{ 'popup-height': type === 'left' || type === 'right' }"><text
						class="text">popup 内容</text></view>
			</uni-popup>
		</view>
	</view>
</template>


<!-- 行为层 -->
<script>
	export default {
		data() {
			return {
				msg: '',
				chatlists: [],
				type: 'center',
				dt_bgc:'f7f7f7'
			}
		},
		methods: {
			deepThink(){
				this.dt_bgc = this.dt_bgc == '#d0d0d0' ? '#f7f7f7':'#d0d0d0'
			},
			newChat(){
					// 所有数据全部初始化
					this.msg = ''
					this.chatlists = []
			},
			sendMsg() {
				// 声明一个变量缓存输入框的数据,方便下面使用：let声明变量
				let msg = this.msg.trim()
				// 必须输入对话内容
				if (msg == '') {
					uni.showToast({
						title: '内容不能为空',
						icon: 'error',
						duration: 1000
					})
					// 不让代码再往下执行
					return false
				}

				// 数据正在处理之中
				uni.showLoading({
					title: '正在思考...',
					mask: true
				})
				console.log(msg)

				// 追加用户提问到对话列表
				this.chatlists.push({
					role: 'user',
					content: msg
				})

				// -------------------------------------------------------------------------
				uni.request({
					url: 'http://localhost:8000/v1/chat/completions', //大模型的访问地址
					data: {
						model: '/models',
						messages: [{
								"role": "system",
								"content": "你是我的个人AI助手，温柔善良知识渊博。"
							},
							{
								"role": "user",
								"content": msg
							}
						],
						"stream": false
					},
					method: 'POST',
					header: {
						'Content-Type': 'application/json' //自定义请求头信息
					},
					success: (response) => {
						let data = response.data.choices[0].message.content
						// 提取推理部分
						let reasoning = data.match(/<think>([\s\S]*?)<\/think>/)?.[1]?.trim() || "";

						// 提取回答部分（去掉前后的think标签内容）
						let answer = data.replace(/<think>[\s\S]*?<\/think>/, "").trim();

						console.log("推理部分：", reasoning);
						console.log("回答部分：", answer);

						// 追加AI回答到对话列表
						this.chatlists.push({
							role: 'assistant',
							content: answer
						})
						// this.chatlists.push({role:'assistant', content:data})
						// 关闭提示框
						uni.hideLoading()
						// 输入框里面的数据需要清空
						this.msg = ''
					}
				});
				// ----------------------------------------------------------------
			},
			change(e) {
				console.log('当前模式：' + e.type + ',状态：' + e.show);
			},
			toggle(type) {
				this.type = type
				// open 方法传入参数 等同在 uni-popup 组件上绑定 type属性
				this.$refs.popup.open(type)
			},
		}
	}
</script>


<style scoped>
	.nav-bar-empty {
		height: 120rpx;
	}

	.message-container {
		display: flex;
		flex-direction: column;
		padding: 16rpx;
	}

	.message-item {
		display: flex;
		margin-bottom: 24rpx;
	}

	.user-message {
		justify-content: flex-start;
	}

	.reply-message {
		justify-content: flex-end;
	}

	.avatar {
		width: 80rpx;
		height: 80rpx;
		border-radius: 50%;
		margin: 0 16rpx;
	}

	.message-content {
		background-color: #e5f2ff;
		/* 用户消息浅蓝色背景，可按需调整 */
		padding: 16rpx;
		border-radius: 20rpx;
		max-width: 560rpx;
		/* 限制消息内容宽度，可按需调整 */
	}

	.wbg {
		background-color: #fff;
	}

	.message-text {
		font-size: 32rpx;
		color: #333;
	}
</style>

<style lang="scss">
	@mixin flex {
		/* #ifndef APP-NVUE */
		display: flex;
		/* #endif */
		flex-direction: row;
	}

	@mixin height {
		/* #ifndef APP-NVUE */
		height: 100%;
		/* #endif */
		/* #ifdef APP-NVUE */
		flex: 1;
		/* #endif */
	}

	.box {
		@include flex;
	}

	.button {
		@include flex;
		align-items: center;
		justify-content: center;
		flex: 1;
		height: 35px;
		margin: 0 5px;
		border-radius: 5px;
	}

	.example-body {
		background-color: #fff;
		padding: 10px 0;
	}

	.button-text {
		color: #fff;
		font-size: 12px;
	}

	.popup-content {
		@include flex;
		align-items: center;
		justify-content: center;
		padding: 30rpx;
		height: 100rpx;
		background-color: #fff;
	}

	.popup-height {
		@include height;
		width: 500rpx;
	}

	.text {
		font-size: 12px;
		color: #333;
	}

	.popup-success {
		color: #fff;
		background-color: #e1f3d8;
	}

	.popup-warn {
		color: #fff;
		background-color: #faecd8;
	}

	.popup-error {
		color: #fff;
		background-color: #fde2e2;
	}

	.popup-info {
		color: #fff;
		background-color: #f2f6fc;
	}

	.success-text {
		color: #09bb07;
	}

	.warn-text {
		color: #e6a23c;
	}

	.error-text {
		color: #f56c6c;
	}

	.info-text {
		color: #909399;
	}

	.dialog,
	.share {
		/* #ifndef APP-NVUE */
		display: flex;
		/* #endif */
		flex-direction: column;
	}

	.dialog-box {
		padding: 10px;
	}

	.dialog .button,
	.share .button {
		/* #ifndef APP-NVUE */
		width: 100%;
		/* #endif */
		margin: 0;
		margin-top: 10px;
		padding: 3px 0;
		flex: 1;
	}

	.dialog-text {
		font-size: 14px;
		color: #333;
	}
</style>



<style>
	.tmp-bottom {
		width: 750rpx;
		height: 180rpx;
	}

	.container {
		display: flex;
		flex-direction: column;
		height: 100vh;
		background-color: #ffffff;
	}

	/* 顶部导航栏 */
	.nav-bar {
		position: fixed;
		left: 0;
		right: 0;
		display: flex;
		flex-direction: row;
		justify-content: space-between;
		align-items: center;
		height: 100rpx;
		padding: 0 30rpx;
		border-bottom: 1rpx solid #f0f0f0;
		background-color: #fff;
		z-index: 99999999;
	}

	.nav-icon {
		width: 40rpx;
		height: 40rpx;
	}

	.nav-title {
		font-size: 32rpx;
		font-weight: bold;
		color: #000000;
	}

	/* 中间主区域 */
	.main-content {
		flex: 1;
		justify-content: center;
		align-items: center;
		text-align: center;
		display: flex;
		flex-direction: column;
	}

	.logo {
		width: 120rpx;
		height: 120rpx;
		margin-bottom: 30rpx;
	}

	.welcome {
		font-size: 36rpx;
		font-weight: bold;
		margin-bottom: 20rpx;
	}

	.desc {
		font-size: 28rpx;
		color: #999999;
		line-height: 48rpx;
		padding: 0 80rpx;
	}

	/* 底部输入区域 */
	.input-area {
		position: fixed;
		bottom: 0;
		left: 0;
		right: 0;
		z-index: 1000;

		width: 690rpx;
		padding: 20rpx 30rpx;
		border-top: 1rpx solid #f0f0f0;
		background-color: #ffffff;
		display: flex;
		flex-direction: column;
	}

	/* 输入框样式 */
	.msg-input {
		width: 610rpx;
		height: 80rpx;
		border-radius: 40rpx;
		background-color: #f7f7f7;
		padding: 0 30rpx;
		font-size: 28rpx;
		margin-bottom: 20rpx;
	}

	.input-placeholder {
		color: #999999;
		font-size: 28rpx;
	}

	.input-options {
		display: flex;
		flex-direction: row;
		align-items: center;
		justify-content: space-between;
	}

	.option {
		display: flex;
		flex-direction: row;
		align-items: center;
		background-color: #f7f7f7;
		border-radius: 30rpx;
		padding: 10rpx 20rpx;
		margin-right: 20rpx;
	}

	.option-icon {
		width: 36rpx;
		height: 36rpx;
		margin-right: 10rpx;
	}

	.option-text {
		font-size: 26rpx;
		color: #333333;
	}

	.send {
		width: 60rpx;
		height: 60rpx;
		border-radius: 50%;
		display: flex;
		justify-content: center;
		align-items: center;
	}

	.send-icon {
		width: 50rpx;
		height: 50rpx;

	}

	.add-icon {
		width: 45rpx;
		height: 45rpx;

	}
</style>