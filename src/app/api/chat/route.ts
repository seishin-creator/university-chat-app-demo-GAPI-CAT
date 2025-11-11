// =========================================================
// 📄 ファイルの役割
// = 1. フロントエンド (Chat.tsx) からのチャットリクエストを受け取るAPIエンドポイントです。
// = 2. Google Geminiモデル (Function Calling対応) を使用して、ユーザーとの会話を処理します。
// = 3. 必要に応じてWeb検索ツール (googleSearch) を呼び出し、リアルタイム情報を提供します。
// = 4. セッションIDに基づき、チャット履歴を管理します。
// = 5. 503エラーおよび429エラー発生時に自動リトライ処理を行います。
// =========================================================
import { NextResponse } from 'next/server';
import { GoogleGenAI, Content, Part } from '@google/genai';
import { generateSystemPrompt } from '@/utils/generateSystemPrompt';
import { google } from 'googleapis';

// 🚨 汎用性を持たせるための定数定義
const AI_NICKNAME = 'CATミュージックカレッジ';
const DEFAULT_ERROR_DESCRIPTION = 'チャットアシスタント'; 

// Chat.tsxからメッセージの型を再定義
type Message = {
    role: 'user' | 'assistant';
    content: string;
};

// Google Custom Search APIクライアントを初期化
const customsearch = google.customsearch('v1');

/**
 * 指定されたミリ秒数だけ処理を一時停止する
 */
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

/**
 * 検索結果をモデルに渡しやすいように整形するヘルパー関数
 */
const formatSearchResults = (searchResults: any[]) => {
    // スニペットを整形するロジックは、GAPI成功ロジックではJSON.stringifyに委ねるため削除
    return searchResults.map((item: any) => ({
        title: item.title,
        snippet: item.snippet,
        link: item.link,
    }));
};

// ★★★ 外部検索ツール（Function Calling）- 実際のWeb検索実装 ★★★
async function googleSearch(query: string) {
    console.log(`🔍 Tool Called! Running Web Search for: ${query}`);

    try {
        const response = await customsearch.cse.list({
            auth: process.env.GOOGLE_SEARCH_API_KEY, 
            cx: process.env.GOOGLE_SEARCH_CX,     
            q: query,                              
            num: 3,                                
        });

        // 検索結果をJSON形式のオブジェクトとして取得
        const searchResults = formatSearchResults(response.data.items || []);

        if (searchResults.length === 0) {
             // 💡 修正: 検索結果なしも JSON オブジェクトとしてラップ (GAPI形式)
             return {
                query: query,
                result: { search_snippet: `検索結果は見つかりませんでした。` },
            };
        }

        // 💡 GAPI成功ロジック: 検索結果のオブジェクト全体を文字列化して返す
        return {
            query: query,
            // 検索結果のオブジェクトをJSON.stringifyで文字列化し、さらに 'search_snippet' でラップ
            result: { 
                search_snippet: `【Web検索結果の抜粋】: ${JSON.stringify(searchResults)}` 
            },
        };
    } catch (error) {
        console.error('Web Search Error:', error);
        return {
            query: query,
            result: { search_snippet: `検索中にエラーが発生しました。Web検索APIでエラーが発生しました。` },
        };
    }
}

// Next.jsのAPIルート (POSTメソッド)
export async function POST(req: Request) {
    const { messages, sessionId } = await req.json();

    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
        return NextResponse.json({ error: 'Gemini APIキーが設定されていません。' }, { status: 500 });
    }

    // 履歴管理の準備
    if (typeof global.chatHistoryMap === 'undefined') {
        global.chatHistoryMap = new Map();
    }
    const chatHistory: Message[] = global.chatHistoryMap.get(sessionId) || [];

    // 現在のメッセージを履歴に追加
    const currentMessage = messages[messages.length - 1];
    chatHistory.push(currentMessage);
    global.chatHistoryMap.set(sessionId, chatHistory);

    // AIのシステムプロンプトを生成
    const systemInstruction = await generateSystemPrompt();

    // Gemini API用の Content 形式に変換
    const contents: Content[] = chatHistory.map(msg => {
        const role = msg.role === 'user' ? 'user' : 'model';
        return {
            role,
            parts: [{ text: msg.content }],
        };
    });

    try {
        const ai = new GoogleGenAI({ apiKey });

        let fullResponse;
        let contentsLength = contents.length; 

        // ★★★ Tool Calling 反復処理とリトライの開始 ★★★
        const maxToolCalls = 5; 

        for (let i = 0; i < maxToolCalls; i++) {
            // 履歴の更新チェックとループ離脱条件
            if (contents.length > contentsLength) {
                contentsLength = contents.length;
            } else if (i > 0) {
                break;
            }

            let response;
            const maxRetries = 5; 
            let currentDelay = 2000; 

            // 💡 リトライロジック: 503/429エラー時に自動再試行
            for (let attempt = 0; attempt < maxRetries; attempt++) {
                try {
                    // 構文修正済み
                    response = await ai.models.generateContent({
                        model: 'gemini-2.5-pro', // 課金が完了するまでは429エラーが発生しやすい
                        contents: contents,
                        config: { 
                            systemInstruction: systemInstruction, 
                            tools: [{
                                functionDeclarations: [
                                    {
                                        name: 'googleSearch',
                                        description: 'リアルタイムのニュース、日付、最新の出来事、一般的なWeb情報など、モデルの訓練データにない外部情報が必要な時に使用する。',
                                        parameters: {
                                            type: 'OBJECT',
                                            properties: {
                                                query: {
                                                    type: 'STRING',
                                                    description: 'Web検索に使用する具体的な検索クエリ（日本語）',
                                                },
                                            },
                                            required: ['query'],
                                        },
                                    },
                                ],
                            }],
                        },
                    });
                    break; // 成功したらリトライループを抜ける
                } catch (error) {
                    let shouldRetry = false;
                    let retryDelayMs = currentDelay;

                    if (error instanceof Error) {
                        const statusCodeMatch = error.message.match(/code\":(\d+)/);
                        const statusCode = statusCodeMatch ? parseInt(statusCodeMatch[1], 10) : 0;
                        
                        if (statusCode === 503) {
                            shouldRetry = true;
                        } else if (statusCode === 429) {
                            const match = error.message.match(/retry in (\d+(\.\d+)?)s/);
                            if (match && match[1]) {
                                retryDelayMs = parseFloat(match[1]) * 1000 + 500; 
                            }
                            shouldRetry = true;
                        }
                    }

                    if (shouldRetry && attempt < maxRetries - 1) {
                        console.warn(`⚠️ API Error (Code: ${error instanceof Error && error.message.includes("code") ? error.message.match(/code\":(\d+)/)?.[1] : 'Unknown'}). Retrying in ${retryDelayMs / 1000}s...`);
                        await delay(retryDelayMs);
                        if (error.message.includes("code\":503")) {
                            currentDelay *= 2; 
                        }
                    } else {
                        throw error; // 最終試行失敗、または非リトライ対象のエラー
                    }
                }
            }

            fullResponse = response;
            const call = response?.functionCalls?.[0];

            // ツール呼び出しが無ければループを抜けて回答を返す
            if (!call) {
                break;
            }

            // ★★★ ツール呼び出しを処理 ★★★
            const functionName = call.name;
            const args = call.args;

            if (functionName === 'googleSearch') {
                const toolResult = await googleSearch(args.query);

                // ツールからの応答を履歴に追加して、モデルに再度送信
                // 💡 GAPI成功ロジック: toolResult.result は JSON オブジェクトを文字列化したもの
                if (response?.candidates?.[0]?.content) {
                    contents.push(
                        response.candidates[0].content, // ツール呼び出しの記述
                        {
                            role: 'function',
                            parts: [{
                                functionResponse: {
                                    name: 'googleSearch',
                                    response: toolResult.result,
                                },
                            }],
                        }
                    );
                } else {
                    break;
                }
            } else {
                throw new Error(`Unknown function call: ${functionName}`);
            }
        }
        // ★★★ Tool Calling 反復処理とリトライの終了 ★★★

        // 最終応答を抽出
        const reply = fullResponse?.text ?? `ごめん、${AI_NICKNAME}はマジでうまく返せへんかったわ😭！`;

        // 成功したら、チャット履歴を最新の応答で更新
        chatHistory.push({ role: 'assistant', content: reply });
        global.chatHistoryMap.set(sessionId, chatHistory);

        return NextResponse.json({ message: reply });
    } catch (error) {
        console.error('❌ API処理中のエラー:', error);
        
        let status = 500;
        let userErrorMessage = `申し訳ありません。サーバーとの通信中に予期せぬエラーが発生しました。`;
        
        if (error instanceof Error) {
            const statusCodeMatch = error.message.match(/code\":(\d+)/);
            const statusCode = statusCodeMatch ? parseInt(statusCodeMatch[1], 10) : 0;

            if (statusCode === 503) {
                status = 503;
                userErrorMessage = '現在サービスが大変混み合っています。リトライを試みましたが解決しませんでした。時間を置いて再度お試しください。';
            } else if (statusCode === 429) {
                status = 429;
                userErrorMessage = '現在、APIの利用制限（クォータ）を超過しています。数分待ってから再度お試しください。';
            } else if (error.message.includes("API_KEY")) {
                 status = 401;
                 userErrorMessage = '認証情報に問題があります。APIキーの設定を確認してください。';
            } else if (statusCode === 400) {
                 status = 400;
                 userErrorMessage = '無効なリクエストが送信されました。アプリケーション側に問題がある可能性があります。';
            } else {
                 userErrorMessage = 'サーバーエラーが発生しました。詳細については、コンソールログを確認してください。';
            }
        }
        
        return NextResponse.json({ error: userErrorMessage }, { status });
    }
}